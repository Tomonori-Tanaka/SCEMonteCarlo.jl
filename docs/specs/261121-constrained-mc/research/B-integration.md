# CMC (Asselin 2010) を SCEMonteCarlo.jl に組み込むための調査報告 (B: コード統合)

調査対象: `../../../../../SCEMonteCarlo.jl`（HEAD `9529572`、作業ツリーはクリーン）。
読み取りのみで、編集・テスト実行はしていない。引用する行番号はすべてこの HEAD のもの。

---

## 0. 結論の要約

- **2 スピン同時移動の ΔE**: 「逐次 leave-one-out」で厳密に計算できる。
  c_i → ΔE_i を出す → `zrows` の列 i に Z(e_i′) を仮に書く → c_j → ΔE_j を出す → 棄却なら列 i を退避コピーから戻す。
  ΔE = ΔE_i + ΔE_j は E(e_i′,e_j′) − E(e_i,e_j) の望遠鏡和なので、体数（body order）によらず厳密になる。
  コストは単一サイト試行 2 回分（`site_coeffs!` ×2、`_zlm_row!` ×2、nlm 長のコピー 1 回）。
  素朴な ΔE_i + ΔE_j が間違うのは i, j が同じインスタンスに入る場合（2 体ボンド、3 体以上の任意のクラスター）。
  逐次法ならこれを分岐なしで吸収できる。
- **並列化**: 受理確率に大域量 M_z が (M_z′/M_z)² の形で入る。このため、インスタンスを共有しない 2 つのペア移動でも可換にならず、色クラス並列（U1）の前提が成り立たない。
  **CMC スイープは直列のみ**とし、並列化は θ 点・温度点（独立レーン、PT レーン）のレベルで行う。
  `sweep_tasks` の serial ≡ parallel 契約は CMC には持ち込まない（新ドライバ側で `sweep_tasks` を受け付けない）。
- **新しい永続状態を最小にできる**: M はスイープ開始ごとに config から固定順序で再計算し、スイープ内だけ増分更新する。
  パートナー j の抽選も移動元 i の `site_rngs[i]` から引く。
  こうすれば `ChainState` にフィールドを足さずに済み、PT の payload 分割テストも checkpoint の chain ブロックも変わらない。
  新たに保存するのは拘束の定義（方向・重み・パートナー規則）という**実行設定**だけ。
- **API**: 新ドライバ `run_cmc(H, constraint::MagnetizationConstraint; …) -> CMCResult` を推奨する（既存 `run_mc` / `run_pt` のシグネチャには触れない）。
  内部では `_compound_sweep!` 系に「移動方式」（mover）を 1 つ通す。
  トルクは `Observable` として `energy_gradient!` 経由で実装する（`cmc_observables(H, c)`）。
- **独立オラクルが強い**: N = 2 スピン系なら CMC の拘束測度を (ρ, φ) の 2 次元求積で厳密に書ける（重み 4ρ、§5 U4）。
  任意の 2 サイト Hamiltonian・任意の T・θ でトルクの厳密値が出る。
  β → 0 では ⟨M_z²⟩ = N（任意の N で厳密）と ⟨M_z⟩ = ⟨|M|⟩_free（N = 2 で 4/3）。
  等方模型では Σ_s G_s × e_s ≡ 0（大域回転不変性）がマシン精度で成り立つ。
  ヤコビアンの各因子を落とすミューテーションは、これらで確実に検出できる。

---

## 1. ペア移動のエネルギー評価

### 1.1 既存カーネルの契約

- `site_coeffs!(c, H, s, zrows)`（`src/energy.jl:100-139`）
  - サイト s に接する全インスタンスについて、他メンバーの tesseral 列（`zrows[:, t]`）を縮約し、c に**加算**する（ゼロクリアはしない）。
  - c は e_s に依存しない。根拠はコンストラクタの「インスタンス内メンバーサイトは相異なる」というアサーション（`src/hamiltonian.jl:294-304`）。
- `delta_energy(c, zold, znew)`（`src/energy.jl:148-157`）は Σ c_k (znew_k − zold_k) を返す。β は含まない。
- `_attempt_metro!`（`src/updates.jl:37-58`）の流れ:
  1. `fill!(sc.c,0)`
  2. `site_coeffs!`
  3. 提案（flip 0.2 / Rodrigues 回転）
  4. `_zlm_row!(sc.znew…)`
  5. `delta_energy`
  6. 受理なら `config[s]`・`zrows[:,s]` を書き、`dE[s] = ΔE` に積む
- RNG 消費契約: 一様乱数は ΔE > 0 のときだけ引く（`docs/specs/updates-stationarity.md` U2）。

### 1.2 素朴な和が誤る理由

c_i も c_j も旧 `zrows` から計算すると、ΔE_i + ΔE_j には次の項が欠ける。

> i と j を同時に含む各インスタンスの ΔZ_i ⊗ ΔZ_j の縮約（交差項）

- 2 体: i–j が直接結合している場合に誤差が出る。
- 3 体以上: i, j がともに入る三体・四体クラスターがあれば、最近接でなくても誤差が出る（tile 後のクラスター半径内）。
- パートナー j はランダムに選ぶので、交差項が出るのは「たまに」になる。統計テストでは見逃しやすい種類のバグなので、決定論的な ΔE ゲートが必須。

### 1.3 推奨する厳密法: 逐次 leave-one-out（tentative column write + restore）

```
fill!(c,0); site_coeffs!(c, H, i, zrows)          # c_i: e_j は旧値
_zlm_row!(zi_new, e_i′, …); ΔE_i = delta_energy(c, zrows[:,i], zi_new)
copyto!(zi_old, zrows[:,i]); copyto!(zrows[:,i], zi_new)   # 仮書き込み
fill!(c,0); site_coeffs!(c, H, j, zrows)          # c_j: e_i′ を見る
_zlm_row!(zj_new, e_j′, …); ΔE_j = delta_energy(c, zrows[:,j], zj_new)
ΔE = ΔE_i + ΔE_j
受理: copyto!(zrows[:,j], zj_new); config[i]=e_i′; config[j]=e_j′; dE[i] = ΔE
棄却: copyto!(zrows[:,i], zi_old)                 # ビット単位で元に戻る
```

**正しさ**: E(e_i′, e_j′) − E(e_i, e_j) = [E(e_i′, e_j) − E(e_i, e_j)] + [E(e_i′, e_j′) − E(e_i′, e_j)]。
括弧内はどちらも単一サイト移動であり、`hamiltonian.jl:294-304` の不変条件のもとで leave-one-out が厳密に成り立つ。
したがって**任意の body order**、i, j が複数のインスタンスを共有する場合、AllImages の像サイトでも厳密。
必要な前提は i ≠ j だけで、ドライバ側で保証する。

**ビットの性質**: i, j がインスタンスを共有しなければ、c_j は列 i を読まない。
このとき逐次法と素朴法の結果はビット単位で一致するので、交差判定の分岐は不要。

**コスト**: `site_coeffs!` ×2 + `_zlm_row!` ×2 + nlm 長のコピー（棄却時はもう 1 回）。
1 CMC スイープ（ペア試行 N 回）は Metropolis 約 2 スイープ分になる。
幾何的な棄却（§1.4 の根号の中身 < 0、M_z′ ≤ 0）は**エネルギー計算の前に**判定し、そこで抜ける。
こうすると棄却されるペアでは `site_coeffs!` が 1 回も走らない。

**却下した代案**:
- (a) 共有インスタンスを列挙して交差項を明示的に加える案は、`site_inst` の交差計算と、fast path（pair / triplet / general）ごとの交差版カーネルが要る。分岐が増え、bitwise 参照ゲートも増えるので割に合わない。
- (b) 旧 SpinClusterMC の `delta_local_energy_pair` を「一発評価」する案は、実質的に (a) と同じ。

**スクラッチ**: `SweepScratch`（`src/state.jl:63-72`）には `c, znew, plm, dE` しかない。
これは public-unexported 層の型で、位置引数コンストラクタを持つ。
フィールドを足さず、内部専用の `_CMCScratch(c, zi_new, zi_old, zj_new, plm)` を新設するのがよい。
`dE` のステージングには既存 `SweepScratch.dE` を流用する。

### 1.4 拘束ジオメトリ（受理則の材料）

- 定義
  - 拘束方向 n̂
  - 重み w_s = w[site_atom(H, s)]（`site_atom`: `src/hamiltonian.jl:549`）
  - M = Σ_active w_s e_s
- 投影: v^⊥ = v − (v·n̂) n̂。フレーム (u, v, n̂) を組まず、デカルト座標のまま書ける。
- パートナー写像（決定論的）:
  - e_j′^⊥ = e_j^⊥ + (w_i/w_j)(e_i^⊥ − e_i′^⊥)
  - e_j′·n̂ = copysign(√(1 − |e_j′^⊥|²), e_j·n̂)（**符号保存**）
  - 根号の中身が ≤ 0 なら棄却（等号も棄却にする: e_jz′ = 0 は比が発散し、逆移動は測度ゼロになる）
- M_z′ = M_z + w_i(e_i′·n̂ − e_i·n̂) + w_j(e_j′·n̂ − e_j·n̂)。M_z′ ≤ 0 なら棄却。
- 受理比: R = (M_z′/M_z)² · |e_j·n̂| / |e_j′·n̂| · exp(−βΔE)
  - 一様乱数は R < 1 のときだけ引く（U2 の消費契約を一般化したもの）。
  - log で比較すると安全: log R = 2 log(M_z′/M_z) + log|e_jz| − log|e_jz′| − βΔE。
- 重み付きでも |e_jz| の比は変わらない。拘束 δ²(Σ w e^⊥) を j について解くと 1/w_j² という定数が出るが、往復比で相殺する。
- **重み比の大きいペアは幾何的にほぼ全棄却になる**。例: Fe（m ≈ 2.2）が動き、Pt 様（m ≈ 0.3）がパートナーの場合。
  - パートナー集合は「i と同じ |w| クラス」（設定に依存しない固定集合）に制限する。設定に依存しないので、詳細釣り合いは保たれる。
  - w = 0 のサイト（moment ゼロだが SCE active）は M に寄与しない。パートナー不要の単一サイト Metropolis として扱う（パートナーにはしない）。
- 符号付き重み: ±1 にすれば反強磁性の Néel ベクトル拘束、副格子ごとに符号を付ければフェリ磁性体の補償点近傍を扱える。同じ式のまま一般化できる。
  `MagnetizationConstraint` の重みを `Vector{Float64}`（cell atom ごと、符号可）として設計しておけば、後から追加 API なしで使える。

### 1.5 ΔE 局所性契約と body ≥ 3

- 契約（CLAUDE.md の「ΔE locality」、`hamiltonian.jl:294-304`）は**サイトごと**の主張であり、逐次法はそれを 2 回使うだけ。新しい不変条件は要らない。
- body ≥ 3 の影響は「交差項の発生頻度」が上がることだけ（i, j を同時に含むインスタンスが増える）。逐次法なら正しさは変わらない。
- 唯一の新しい前提は i ≠ j。パートナー抽選は「i を除く候補集合から 1 回だけ引き、i 以降の添字をずらす」形にする。棄却ループを作らず、RNG 消費を一定にするため。

---

## 2. 決定論と並列性

### 2.1 既存の契約

- U1（`updates-stationarity.md`、`src/updates.jl:1-16, 157-206`）
  - 色クラス内のサイトはインスタンスを共有しないので、単一サイトカーネルが可換。
  - `site_rngs` はサイトごと（`src/state.jl:43`）。
  - ΔE は `dE[s]` にステージングし、色順の `_reduce_dE`（`src/updates.jl:92-98`）で集約する。
  - 以上により、`sweep_tasks` の数によらずビット一致する（`test_parallel.jl`）。
- P3（PT）: レーン RNG はマスターから固定順で導出し、交換用の一様乱数は事前に引いておく。

### 2.2 ペア移動は色クラス並列に乗らない

受理比に大域量 M_z が入る。
2 つのペア (i, j) と (k, l) がインスタンスを全く共有しなくても、片方の受理が M_z を変え、もう片方の受理確率が変わる。
つまり可換性が成り立たず、「同時実行 ≡ ある直列順」という U1 の論拠が崩れる。
厳密な並列化の手段はない（推測的に ΔE を並列計算して受理だけ直列にする手は、効果に見合わない）。

**推奨**:
- CMC スイープは**直列のみ**（v1）。
- `run_cmc` には `sweep_tasks` キーワードを置かない。checkpoint には `plan/sweep_tasks = 1` を書く。
- 並列化の単位は次の 2 つ。
  - **θ 点ごとの独立チェーン**（PT から交換を抜いたレーン機構。マスター → レーン RNG を固定順で導出し、`ntasks` 非依存ゲートを置く）
  - **温度ラダーの PT レーン**
- 異方性の K(T) は θ × T のグリッドで測るので、この粒度で十分に並列化できる。

将来の選択肢として**ブロック拘束**がある。セル（ブロック）ごとに M_B を拘束し、パートナーを同じブロック内に限る。
受理比は自ブロックの M_B だけに依存するので、インスタンスを共有しないブロック同士は可換になり、色付けで並列化できる。
ただし**アンサンブルが変わる**（拘束が強くなる）。性能目的で黙って採用してはならない。

### 2.3 スイープの定義と RNG 消費（提案）

- 1 CMC スイープ = 対象となる各移動元 i（active かつ w_i ≠ 0）について 1 回のペア試行。i は `H.color_sites` の順に走査する。w_i = 0 の active サイトは単一サイト試行とする。
  - Asselin の「1 MCS = N 回の試行ペア移動」に相当する。
  - サイト選択に RNG を使わない点は U1 と同じ。
- 色順で走査するのは、`dE[i]` のステージングと `_reduce_dE` をそのまま使うため。直列なので色順である必要は本来ない。i はスイープ内で 1 回しか移動元にならないので、`dE[i] = ΔE_pair` という代入で衝突しない（j のスロットには書かない）。
- RNG はすべて `st.site_rngs[i]` から引く。順序:
  1. パートナー添字（1 回）
  2. flip/回転を選ぶ一様乱数
  3. 回転軸（`_random_unit`、randn ×3 以上）
  4. 回転角（randn）
  5. （R < 1 のときだけ）受理用の一様乱数
  - チェーン RNG（`st.rng`）はスイープ内で消費しない。
  - `site_rngs` は既に checkpoint に保存されている（`checkpoint.jl:169`）ので、**新しい RNG 状態はゼロ**。
- 幾何的な棄却も試行に数える（`att_metro += 1`）。この場合、受理用の乱数は引かない。
- 定常性の議論:
  - i を固定したカーネル K_i は、j を設定非依存の固定集合から一様に選び、提案は対称、パートナー写像は決定論的で符号を保存する（逆移動は同じ (i, j) で元に戻る）。
  - よって K_i は拘束測度に関して可逆。
  - 固定順の合成は定常（U1 と同じ論法）。

### 2.4 適応ステップとの相互作用

- カウンタは `acc_metro` / `att_metro` を**流用**するのが最小構成。CMC では単一サイト Metropolis を混ぜられない（拘束を壊す）ので、意味が衝突しない。
  - `_adapt_step!`（`src/updates.jl:296-305`）は変更不要。thermalization のみ、境界で凍結。
  - `TempResult.acceptance_metropolis` の意味は「ペア受理率」になる。docstring で明記する。
  - 専用カウンタを足すと `ChainState` にフィールドが増える（下記 §3.1 の波及あり）ので推奨しない。
- 受理率はステップに対して単調。小さなステップほど ⊥ 変化が小さく、幾何的な棄却も減るので、target 0.5 で動く。
- U3 の注意（自由サイトがあると受理率に床ができる）の CMC 版: β → 0 でも幾何的な棄却があるので受理率は 1 にならない。適応のゲートは結合した模型で行う。
- flip 提案（`_FLIP_FRACTION = 0.2`、`updates.jl:20`）について
  - 対称なので正しい。ただし ⊥ 成分が 2|e_i^⊥| 跳ぶため、パートナーの根号が負になりやすい。
  - CMC は M を固定しているので、ローブ間エルゴード性は主目的ではない。
  - **回転のみ（flip 0）を推奨**するが、提案の選択はユーザー確認事項とする（§4.3）。

---

## 3. 触れる既存契約・連動箇所の全リスト

### 3.1 `ChainState` / PT payload / checkpoint chain ブロック（触れない設計を推奨）

- `ChainState`（`src/state.jl:17-33`）にフィールドを足した場合の波及:
  - (a) 位置引数コンストラクタ呼び出し 2 か所: `checkpoint.jl:189`、`checkpoint.jl:600-604`（gpu resume）
  - (b) PT の fieldnames 完全分割テスト（`test/unit/test_pt.jl:169-183`）。payload か lane 所属かの分類が必要
  - (c) `_write_chain` / `_read_chain`（`checkpoint.jl:165-192`）
  - (d) GPU 側のミラー
- **M をスイープ開始ごとに再計算する設計**ならこれらを全部避けられる。
  - 再計算は O(N) の SVector 加算で、スイープ（N × 隣接 × エントリ）に比べて無視できる。
  - スイープ内だけ増分更新する。
  - スイープ境界では M の増分値が存在しないので、checkpoint への保存も、PT スワップでの受け渡しも不要。
  - スワップ後の次スイープで自動的に正しい M になる。
- **罠**: M を lane や ChainState の外にキャッシュして持ち越すと、`_swap_payload!`（`pt.jl:59-64`）が config/zrows/energy しか交換しないため、スワップ後に他人の M_z で受理判定してしまう。

### 3.2 Overrelaxation（`_attempt_or!`、`updates.jl:66-87`）

- 反射 e′ = 2(e·ĥ)ĥ − e は e^⊥ を変えるので**拘束を壊す**。
- 「反射 + パートナー補正」は、j が i の近傍にいると逆移動で軸 ĥ_i が変わり、involution にならない。正しい受理則を導出するには別途ヤコビアンの議論が要る。
- **v1 では対象外**とし、`run_cmc` に `or_per_metropolis` を置かない（`_resolve_or_passes`、`run.jl:131-139` も経由しない）。
- `run_pt` / `run_mc` に constraint キーワードを足す案を取る場合は、`or_per_metropolis > 0` を door で拒否する必要がある。

### 3.3 適応ステップ

§2.4 のとおり変更不要（カウンタ流用）。

### 3.4 PT（`run_pt`、`pt.jl`）

- **スワップ則は不変**。CMC の目標測度は π_β(x) ∝ e^{−βE} M_z² δ²(M^⊥) θ(M_z)。
  同じ n̂・同じ重みの 2 レーン間では、M_z² とデルタ関数が β に依存しないので比で相殺し、
  `_swap_accepts`（`pt.jl:92-96`）の min(1, exp(Δβ ΔE)) がそのまま正しい。
  よって CPU/GPU 共有の 1 関数を変更する必要はない。
- 前提: **全レーンで拘束（n̂、重み）が同一**であること。ドライバが 1 つの constraint を全レーンに配る形にすれば、構造的に保証される。
- 罠 1: レーン初期化 `_initial_config(H, init, lane_rngs[r])`（`pt.jl:420`）は init = nothing のとき一様乱数配置を返し、拘束を満たさない。CMC では拘束付きの初期化に置き換える（§3.11）。
- 罠 2: `_lane_segment!`（`pt.jl:69-86`）は `_compound_sweep!` を直接呼ぶ。移動方式（mover）をここまで通す必要がある。
- 罠 3: 観測量の関数 `Observable.f` は**全レーンで共有され**、`_pt_block_async!`（`pt.jl:203-222`）で並行に呼ばれる。可変なスクラッチを捕捉したクロージャ（たとえば事前確保した G バッファを持つトルク観測量）は**データ競合**になる。トルク観測量は純関数で書く（§3.5）。

### 3.5 観測量・binning

- `Observable(name, ncomp, f(config, energy, H))`（`observables.jl:19-28`）は `zrows` を受け取らない。
- トルクは公開 API `energy_gradient!`（`energy.jl:235-252`）から T = Σ_s G_s × e_s として計算する。
  - 符号の検算（一軸 E = −K(e·ẑ)²、e = (sin θ, 0, cos θ)）: T = (0, −K sin 2θ, 0)。
    回転軸は ŷ（ẑ × x̂）、F = −K cos² θ なので dF/dθ = K sin 2θ。
    よって T·ŷ = −dF/dθ で、課題記述の規約と一致する。
  - docstring（`energy.jl:217-218`）の「LL トルク τ_s = G[s] × e_s」の総和と同じものになる。
- コスト: `energy_gradient!` は内部で `_zrows(H, config)`（`energy.jl:240`）を**毎回確保して再構築**する。測定 1 回で約 1〜2 スイープ分かかる。
  - v1 は `measure_interval` で吸収し、割り当ては許容する。前例として `_sublattice_m` も毎回確保している（`observables.jl:131`）。
  - 必要になれば、チェーンの `st.zrows` を使う内部パス（`minimize.jl:54` の `_gradient!` は既存 `zrows` を受け取る）にする。ただし Observable の抽象を迂回することになる（§4 の選択肢）。
- **正規化（規約の決定事項）**: トルクは示量的。
  - `:M` と同じ「訓練セルあたり」（÷ n_cells = n_sites ÷ n_cell_atoms、`observables.jl:112`）を推奨する。dims をまたいで比較でき、K は eV/訓練セル になる。
  - `Evaluable` の f は `n_active` しか受け取らない（`observables.jl:160-166, 244`）ので、セルあたりへの変換は Observable 側で行う。
- 提案する観測量セット `cmc_observables(H, c)`:
  - `:energy`、`:energy2`
  - `:torque`（3 成分、訓練セルあたり）
  - `:torque_rot`（スカラー = T·ê_rot、θ スキャンの回転軸 ê_rot がある場合）
  - `:M_par`（M·n̂、拘束量の重み付き射影）
  - `:constraint_residual`（|M^⊥|/|M|、ドリフト診断。Observable にすれば新しい状態が要らない）
- `standard_evaluables` の χ と U は、拘束下では意味が変わる（|M| のゆらぎのみ）。CMC の既定は `:specific_heat`（拘束付き）だけにする。
- `binning-observables.md` B3 は**規約の唯一の正本**なので、トルクの定義と正規化を追記する必要がある（確認事項）。
- **対称性による零チェック**: ⟨T·n̂⟩ = 0（n̂ まわりの大域回転で拘束アンサンブルは不変）。誤差棒内で 0 になることを観測量ゲートとして使える。

### 3.6 Inactive サイト

- 規約（CLAUDE.md の「Inactive-site convention」）: 移動元・パートナーにしない、M から除外、ビット単位で凍結。
- 3 つすべてを守る必要がある。
- パートナー候補集合を構築するときに `site_active` でフィルタする（`hamiltonian.jl:382`）。
- `_renormalize!` が inactive を飛ばす挙動（`state.jl:170-171`）はそのままでよい。

### 3.7 Zeeman / `magmoms`

- `weights = :magmoms` のときは `H.magmoms`（`hamiltonian.jl:264`）を cell atom 順に読む。`magmoms === nothing` なら door で拒否する。
- 磁場があるとトルクに Zeeman 部分 μ_B M × B が入る。`energy_gradient!` に既に含まれている（`energy.jl:221-224`）ので、物理的にもそれで正しい。手で足すと二重計上になる。
- 既存の `:M`（`observables.jl:104-113`）は Σ m e / n_cells。重み付き拘束の M と定義が同じなので流用できる（`:M_par = M·n̂`）。

### 3.8 再正規化（`_renormalize!`、`state.jl:168-185`）

- normalize による O(eps) の変化で、M^⊥ は O(eps·√N) だけずれる。
- パートナー写像の丸め誤差は、試行回数に対してランダムウォークで蓄積する。1e10 試行でも |M^⊥| は約 1e-11 で、|M| ≈ N·w に比べて無視できる。
- **再射影（剛体回転で M̂ を n̂ に戻す）は推奨しない**。軌道のビットが変わり、異方的な模型ではエネルギーも O(δ) 変わる。
- 代わりに次の 2 点で対応する。
  - スイープ開始時の M 再計算
  - `:constraint_residual` 観測量 + 閾値での `@warn maxlog=1`（`_renormalize!` のドリフト警告と同じ形）
- `_renormalize!` 自体は変更しない（共有コード）。

### 3.9 checkpoint スキーマ + `_fingerprint`

- `_fingerprint`（`checkpoint.jl:76-114`）は**変更しない**ことを推奨する。拘束は模型の同一性ではなく実行設定（kts と同じ層）。変更すると全 fingerprint が変わり、dependent package にも波及する。
- 新しい kind を加える（`"cmc"`、将来 `"cmc_pt"`）。`"gpu_pt"` の前例（`checkpoint-schema.md`、追加的、旧版は unknown kind で明示的に失敗）に倣えば、スキーマ版数を上げずに済む。
  - `constraint/{direction(3), weights(n_cell_atoms), partner_rule(String), rotation_axis(3) or empty}` の group を追加する。
  - resume は拘束をファイルから読む。呼び出し側に再指定させないので、不一致チェックも要らない。
- **罠（resume の既定値）**: `resume(path, H; observables = standard_observables(H), …)`（`checkpoint.jl:458-462`）は、kind を読む前に既定値を評価する。
  CMC の checkpoint を既定のまま resume すると、観測量名の不一致エラー（`checkpoint.jl:478-481`）になる。
  `observables = nothing` を既定にし、kind と拘束を読んでから `cmc_observables` を組み立てる変更が必要。これは公開シグネチャの変更なので確認事項。
- resume ゲートの規律: `_mc_loop!` は温度境界で無条件に書き込む（`run.jl:228-229`）。そのため CMC の resume ゲートも、`_poison_pair`（`test/unit/test_checkpoint.jl:21-27`）で計測途中に中断させ、ファイルの位置を assert する必要がある。

### 3.10 GPU

- **対象外を推奨**。デバイスカーネルは色並列の単一サイトで、Philox のスロット配置・G2–G4 の bitwise 契約がある。大域 M_z の結合はこれと根本的に合わない。
- GPU を使う道があるとすれば「θ × T の独立チェーンをデバイス上に並べる」形だが、別スペックにすべき。

### 3.11 初期配置・`carryover = false`

- 罠: `_initial_config(H, nothing, rng)`（`state.jl:90-92`）は一様乱数配置を返し、拘束を満たさない。
- 罠: `_mc_loop!` の `carryover = false` 分岐（`run.jl:217-220`）が温度ごとに乱数配置で再起動する。
- CMC では、ユーザーのパターン（既定: 全スピンを n̂ 方向、符号付き重みなら sign(w)·n̂、`init` 指定時はその配置）を**剛体回転して M̂ を n̂ に合わせる**。
  - |M_init| ≈ 0 なら door で拒否する（反強磁性で一様重みの場合など）。
  - 拘束サイトが 2 未満なら拒否する（パートナーが存在しない）。
- 再起動も同じ決定論的な初期化で行い、RNG を消費しない。

### 3.12 `reduce_cell`

- 直交していて影響はない。CMC は任意の `TiledHamiltonian` 上で動く。
- 重みは H の cell atom 順（= `site_atom`）で与える。縮約セルを使う場合は縮約後の原子順になる、と docstring に一文書けば足りる。

### 3.13 `minimize_energy` / `find_ground_state`

- T → 0 の参照値として「拘束付き基底状態」を作るには、M^⊥ = 0 の接空間への射影と拘束多様体へのレトラクションを持つ降下法が要る（`minimize.jl:96` の `_minimize!` は積球面のみ）。**v1 では対象外**。
- 代わりの方法:
  - 無拘束の基底状態を剛体回転した E(θ)（上界であり、J ≫ K ならカンティングは O(K²/J)）
  - 低 T の CMC
- `find_ground_state` の出力を CMC の `init` に渡し、各 θ へ剛体回転する、という使い方はそのまま成立する。

### 3.14 その他の小さな罠

- `metropolis_sweep!` の `att_metro += H.n_active`（`updates.jl:168`）は「1 スイープ = active サイト数の試行」を前提にしている。CMC は実試行数（移動元の数）を数える。
- `_print_points_table`（`run.jl:102-116`）の `acc` 列はペア受理率になり、χ・U 列は CMC 既定の観測量では "-" になる（害はない）。
- パートナー抽選に `rand(rng, 1:n)` を使うと、内部で可変個の UInt64 を消費しうる。状態が決まれば決定論的なので問題ない。Julia の版をまたいだ不変性は P6 で既に保証外。
- `TempResult` は θ の情報を持たない。CMC の結果型で外側に持たせる（§4）。

---

## 4. API 設計案

### 4.1 選択肢

**A. `run_mc` / `run_pt` に `constraint = nothing` キーワードを足す**
- 長所
  - ドライバのループ・checkpoint・PT 機構をそのまま流用できる。
  - 入口が 1 つで済む。
- 短所
  - `or_per_metropolis` と `sweep_tasks` が条件付きで不正になる（door が増える）。
  - `run_mc` の docstring に書いた「`sweep_tasks` によらずビット一致」の約束に例外ができる。
  - 既定の観測量（`standard_observables`）が CMC 向きでない。
  - `MCResult` に θ や拘束の情報がない。
  - 既存のエクスポート関数のシグネチャ変更になる。
  - kind `"mc"` に任意 group が混ざり、reader が分岐する。

**B. 新ドライバ `run_cmc(H, c::MagnetizationConstraint; temperature | kT, …) -> CMCResult`（推奨）**
- 長所
  - 既存 API を変えず、追加だけで済む。
  - CMC に固有の既定値を持てる（`cmc_observables`、OR なし、`sweep_tasks` なし、拘束付き初期化）。
  - 結果型が拘束を持てる。
  - checkpoint kind を分離できる。
- 短所
  - 内部ドライバの共通化（mover を通す）が要る。これは層をまたぐリファクタなので「実装前に提案」が必要。

**C. θ スキャンを 1 呼び出しで扱う `run_cmc(H, c; angles = θs, plane = (a, b), …)`**
- 独立レーンをスレッドで並列に走らせる。
- 長所: 主用途（K(T)）に直結し、並列性も自然。
- 短所: レーン機構と checkpoint（レーン構造）が v1 から必要になる。

### 4.2 推奨構成

- `MagnetizationConstraint(direction; weights = :uniform | :magmoms | Vector{Float64}, partner = :same_weight)`
  - 不変の値オブジェクト。符号付き重みを許す（Néel / フェリ）。将来は `blocks` で拡張できる。
- `run_cmc(H, c; temperature | kT, sweeps_therm, sweeps_measure, measure_interval, step, adapt_target, adapt_interval, renorm_interval, nbins, observables = cmc_observables(H, c), evaluables, init, carryover, seed, checkpoint, checkpoint_interval) -> CMCResult`
  - キーワードは `run_mc` から `or_per_metropolis` と `sweep_tasks` を除いたもの。
  - 温度ラダーは `run_mc` と同じ（`carryover` による焼きなまし）。
- `CMCResult(points::Vector{TempResult}, constraint, final_config, seed)`
- θ スキャン（第 2 段）: `cmc_scan(H, plane_axes, θs; …)` または `run_cmc(H, [c₁, …]; ntasks)`
  - 独立レーンで、`ntasks` によらずビット一致することをゲートにする。
  - 結果は θ × T。
  - ê_rot = normalize(a × b)、n̂(θ) = cos θ a + sin θ b を規約として固定する。
- K(T) ヘルパ（第 3 段、任意）
  - トルクの θ 点から、一軸の F = K₁ sin²θ + K₂ sin⁴θ を重み付き最小二乗で当てる（−dF/dθ = −K₁ sin 2θ − 2K₂ sin²θ sin 2θ）。
  - または台形則で F(θ) を積分し、誤差を伝播させる。
  - Asselin の θ = π/4 一点法（⟨T_rot⟩ = −(K₁ + K₂)）を docs の例にする。
- PT（第 4 段）: `run_cmc_pt` か、`run_cmc(…; temperature = ladder, exchange_interval)` で PT を選ぶ形。スワップ則は不変（§3.4）。
- 公開範囲: `total_torque(H, config)` は public-unexported（`energy_gradient` と同じ層）。`run_cmc`、`CMCResult`、`MagnetizationConstraint`、`cmc_observables` はエクスポート候補。

### 4.3 CLAUDE.md の「必ず確認」に該当する変更（一覧）

1. **公開 API の追加・変更**
   - `run_cmc` / `CMCResult` / `MagnetizationConstraint` / `cmc_observables`（エクスポート）
   - `total_torque`（public）
   - `resume` の既定値 `observables = nothing` 化と、戻り値の Union への `CMCResult` 追加
   - 選択肢 A を取る場合は `run_mc` / `run_pt` のシグネチャ変更
2. **checkpoint スキーマ**: kind `"cmc"`（将来 `"cmc_pt"` / レーン構造）と `constraint/` group の追加。追加的として v3 のまま行くか、v4 にするかの判断を含む。
3. **`_fingerprint`**: 変更しない方針自体を確認する（拘束を fingerprint に入れないこと）。
4. **物理規約**
   - トルクの符号: T = Σ G × e = −Σ e × ∂E/∂e、⟨T·ê_rot⟩ = −dF/dθ
   - 正規化: 訓練セルあたり
   - θ と回転軸の規約
   - 重み付き拘束の定義（M = Σ w e、`:magmoms` の場合は μ_B 単位）
   - `binning-observables.md` B3 への追記
5. **詳細釣り合いの新規則**: Asselin の受理比（M_z² と |e_jz| のヤコビアン）、パートナー集合の規則（same-weight クラス）、CMC での flip 提案の有無。
6. **決定論契約の範囲**: CMC は直列のみ（`sweep_tasks` 非対応）であることを契約文に書く。θ レーンの `ntasks` 非依存、resume ≡ uninterrupted は維持する。既存契約を緩めるわけではないが、選択肢 A を取る場合は `run_mc` の約束に例外ができるので確認が要る。
7. **層をまたぐリファクタ（実装前に提案）**: `_compound_sweep!` / `_run_temperature!` / `_mc_loop!` / `_lane_segment!` に mover を通す変更。既存経路のビット中立性を pin で示す必要がある。
8. スペックフォルダの作成（複数日、複数の設計判断）と新しい決定記録 `docs/specs/constrained-mc.md`。`updates-stationarity.md` に CMC のペアカーネルの定常性の節を足すか、新しい記録に置くかも決める。

---

## 5. 段階分割（依存順、各単位 = 1 ファイルまたは 1 つのまとまった機能）

| # | 単位 | 内容 | ゲート（オラクルは実装から独立） |
|---|---|---|---|
| U0 | spec + 決定記録の草案 | `docs/specs/YYMMDD-constrained-mc/` と `constrained-mc.md` の骨子。拘束測度 π ∝ e^{−βE} M_z² δ²(M^⊥) θ(M_z) の導出（重み付き版を含む）、ペア ΔE、RNG 消費、トルクの規約と正規化、対象外（GPU / OR / 拘束付き minimize / ブロック）。§4.3 の確認事項をここで合意する | spec-reviewer |
| U1 | ペア ΔE カーネル（`energy.jl` 末尾、または新規 `src/constraint.jl`） | `_pair_delta_energy!`（逐次 leave-one-out + 仮書き込みと復元） | (a) ΔE_pair と `total_energy` の差が 1e-12 で一致。i, j がインスタンスを共有しない / 2 体ボンドを共有 / triplet fast path（`fixtures.jl:109-117`）/ general path（`fixtures.jl:122-131`、N₁ ≥ 4）/ Zeeman テンプレートを含む場合。(b) 共有ケースで**素朴和が失敗する**ことを示し、ゲートが空でないことを確認。(c) 棄却後に `zrows` がビット単位で元に戻る（`==`）。(d) i ≠ j の door |
| U2 | トルク（`energy.jl` または `observables.jl`） | `total_torque(H, config)` = Σ_active G × e（`energy_gradient!` 経由） | (a) 等方 Heisenberg（`_chain_terms` 等）で任意の配置について T ≡ 0（大域回転不変性、マシン精度）。(b) 小さな剛体回転 R(φ, ê) の `total_energy` 中心差分で dE/dφ = −T·ê。(c) 一軸 body-1 l=2 の手組み項について、閉形式 −K sin 2θ の符号と大きさ。(d) Zeeman 項で −μ_B Σ m e × B の手計算 |
| U3 | 拘束の値型 + 幾何（新規 `src/constraint.jl`） | `MagnetizationConstraint`、重みの解決（`:magmoms` を含む）、パートナー候補集合（active かつ同じ |w| クラス、w = 0 は除外）、パートナー写像、受理比の純関数、剛体回転による初期化 | (a) 往復写像 (e_i → e_i′ → e_i) で e_j が戻る（1e-15）。(b) M^⊥ の保存（1e-14·N）。(c) 符号保存。(d) door 群: 拘束サイト < 2、|M_init| ≈ 0、`magmoms` なしでの `:magmoms`、非有限な重み |
| U4 | CMC スイープ（新規 `src/cmc.jl` 推奨。`updates.jl` は既存契約の正本なので混ぜない） | `cmc_sweep!(st, H, β, c, sc)`: 直列、色順の移動元走査、`site_rngs[i]` の消費順、幾何的棄却を先に判定、`dE[i]` へのステージング + `_reduce_dE`、スイープ開始時の M 再計算、カウンタ流用 | (a) **β → 0 の幾何測度**: ⟨M_z²⟩ = N（任意の N で厳密）、N = 2 で ⟨M_z⟩ = 4/3（p(ρ) ∝ ρ から導出。N ≥ 3 は iid の単位ベクトル直接サンプリングという独立実装を参照）。M_z² 因子を落とす / |e_jz| 比を落とすミューテーションを検出できることを、許容幅（測定 σ × 余裕）とあわせて記録。(b) **N = 2 の厳密求積**: e₁ = ρ(cos φ u + sin φ v) + √(1−ρ²) n̂、e₂ = −ρ(…) + √(1−ρ²) n̂、Z(θ) = ∫₀¹ 4ρ dρ ∫₀^{2π} dφ e^{−βE}。任意の 2 サイト E（一軸 + 交換を閉形式で書く）について ⟨T_rot⟩ = −dF/dθ と ⟨E⟩ を有限 T で比較（dims = (2,1,1) の鎖）。(c) ⟨T·n̂⟩ = 0。(d) 等方 Heisenberg、有限 T で CMC の ⟨M_z⟩ と無拘束 `run_mc` の ⟨|m|⟩·n_active が一致（等方性の恒等式）。(e) 増分エネルギーと再計算値のドリフト、|M^⊥| の上限。(f) 同じ seed でビット一致 |
| U5 | ドライバ（`run.jl` の共通化 + `src/cmc.jl` 内の `run_cmc`） | mover を `_compound_sweep!` / `_run_temperature!` / `_mc_loop!` に通す。`carryover = false` の再起動の差し替え、`CMCResult`、`cmc_observables` | (a) **既存経路のビット中立性 pin**: `run_mc` / `run_pt` の固定 seed 結果がリファクタ前と `==`（変化検出器と明記、取得時点を記載）。(b) 低 T の一軸強磁性鎖で ⟨T_rot⟩(π/4) → −K（J ≫ K、kT ≪ J の漸近。許容幅は σ + 1 次補正の上限）。(c) 等方模型で ⟨T_rot⟩ = 0。(d) door（OR / `sweep_tasks` を受け付けない、`init` の拘束化） |
| U6 | checkpoint（`checkpoint.jl`、`checkpoint-schema.md`） | kind `"cmc"`、`constraint/` group、`resume` での分岐と既定観測量の遅延解決 | (a) resume ≡ uninterrupted を計測途中で確認（`_poison_pair`、位置を assert）。(b) 拘束の round-trip。(c) 誤った kind / 誤った H でのエラー。(d) 既存 kind のファイルが従来どおり読める |
| U7 | θ スキャンレーン（任意で第 2 段） | 独立レーンをスレッドで並列実行（PT から交換を抜いた構成）、θ × T の結果 | `ntasks = 1` ≡ `ntasks = R` の `==`、resume |
| U8 | 拘束付き PT | `run_cmc` の PT モード（または `run_cmc_pt`）。全レーンで拘束が同一、拘束付きの初期化 | スワップ則が不変であることの確認（payload 分割テストは**変更なし**であること自体がゲート）、`ntasks` 非依存、resume |
| U9 | K(T) ヘルパ + docs | torque → K₁, K₂ の当てはめ（または F(θ) の積分）、guide ページ、theory、SPEC / README、CHANGELOG | 合成データ（既知の K₁, K₂ から作ったトルク）での回収。docs のビルド |
| — | 4 軸レビューパネル | U5〜U6 着地時と全体着地時 | — |

**対象外（明記する）**: GPU、拘束付き overrelaxation、拘束付き `minimize_energy`、ブロック拘束、`sweep_tasks` によるペア並列化。

### 現行コードに潜む罠（まとめ）

1. 「1 スイープ = 単一サイト試行 × n_active」の前提（`updates.jl:168`、`att_metro`）。
2. `_attempt_metro!` は受理時に即座に書き込む（`updates.jl:51-55`）。ペア移動では列 i の仮書き込みと棄却時の復元が必要。`SweepScratch` には退避用のバッファがない。
3. 乱数による初期配置（`state.jl:90-92`）と、`carryover = false` の乱数再起動（`run.jl:217-220`）、PT レーンの乱数初期化（`pt.jl:420`）は、どれも拘束を破る。
4. PT の `_swap_payload!` は config/zrows/energy しか運ばない。CMC 固有のキャッシュを持ち越すと壊れる（→ スイープごとの再計算で回避）。
5. 観測量のクロージャは PT レーン間で並行に共有される（`pt.jl:81`）。可変スクラッチを捕捉してはならない。
6. `resume` の既定観測量が kind を読む前に評価される（`checkpoint.jl:459`）。
7. `energy_gradient!` は呼ぶたびに `zrows` を確保して再構築する（`energy.jl:240`）。トルク測定のコストになる。
8. Evaluable は `n_active` しか受け取らない（`observables.jl:244`）。セルあたりの正規化は Observable 側で行う。
9. OR の反射は拘束を壊す。`_resolve_or_passes` の door は「l = 1 チャネルの有無」しか見ていない。
10. `_renormalize!` は M^⊥ に O(eps) のずれを入れる（再射影はせず、再計算 + 診断で対処）。
11. flip 提案 0.2 は CMC では幾何的棄却を増やす。
12. 重み比の大きいパートナーは幾何的にほぼ全棄却になる（same-weight クラスに制限する）。

---

## 6. SpinClusterMC の設計メモ（`SpinClusterMC.jl/docs/specs/260512-simple-impl/design.md:76-101, 393`）はどこまで使えるか

| 旧案 | 判定 | 理由 |
|---|---|---|
| `delta_local_energy_pair` を追加（`design.md:95`）/ buffer pattern（`design.md:93-94`） | **使える（中身は buffer pattern）** | 旧案 (a)「spins を変えてから j を評価し、巻き戻す」がまさに逐次 leave-one-out。ここでは buffer = `zrows` の列 i（退避は nlm 長のコピー 1 つ）。(b) の「一発評価」は交差項処理が要るので採らない。関数名は `_pair_delta_energy!` などの内部名でよい |
| MC 型に optional フィールド `constraint::Union{Nothing,ConstraintState}`（`design.md:86`、`design.md:393`） | **そのままでは使えない** | 本パッケージの `ChainState` は public 層の可変 struct で、位置引数コンストラクタ（`checkpoint.jl:189, 600`）、PT の payload 完全分割テスト、checkpoint の chain ブロックに縛られている。拘束の定義は不変な値としてドライバに渡し、派生量 M はスイープごとに再計算する設計の方が、波及がゼロで済む |
| Lagrange 乗数 λ などの追加状態（`design.md:85`） | **不要** | Asselin の CMC はペア補正 + ヤコビアンで厳密に拘束する。λ は別手法（ペナルティ法・拘束付き最小化）のもの。拘束付き minimize を将来やる場合にだけ関係する |
| 観測量: 拘束方向のトルク、`measure!` の分岐（`design.md:98-99`） | **考え方は使える、実装方法は変える** | 本パッケージの観測量は合成可能な `Observable` なので、分岐は不要。`energy_gradient!`（旧案のいう `gradient` API が既にある）の上に `Observable` を積むだけでよい。ただし PT での共有クロージャと `zrows` の再構築コストという、新しい罠がある |
| 「energy API が揃っていれば足場になる」（`design.md:100-101`） | **使える** | `site_coeffs!` / `delta_energy` / `energy_gradient!` の 3 つで、ペア ΔE とトルクの両方が書ける。カーネル（`energy.jl`）への変更はゼロでよい |
| renorm は「Σ S = const を保つため」の保険（`design.md:430-433`） | **逆向きに注意** | 本パッケージの `_renormalize!` は拘束をわずかに**壊す**側（O(eps)）。スイープ開始時の M 再計算と残差の観測で扱う |

---

## 7. 参照した主なファイルと行番号

- `src/energy.jl`
  - 100-139 `site_coeffs!`
  - 148-157 `delta_energy`
  - 190-205 `_site_grad`
  - 235-252 `energy_gradient!`（240 で `_zrows` を毎回確保）
- `src/updates.jl`
  - 20 `_FLIP_FRACTION`
  - 37-58 `_attempt_metro!`
  - 66-87 `_attempt_or!`
  - 92-98 `_reduce_dE`
  - 157-206 スイープ（168 `att_metro += n_active`）
  - 296-305 `_adapt_step!`
- `src/state.jl`
  - 17-46 `ChainState`
  - 63-72 `SweepScratch`
  - 90-104 `_initial_config`
  - 154-162 `_reset_config!`
  - 168-185 `_renormalize!`
- `src/run.jl`
  - 5-51 `UpdatePlan`
  - 131-139 `_resolve_or_passes`
  - 143-150 `_compound_sweep!`
  - 156-205 `_run_temperature!`
  - 210-232 `_mc_loop!`（217-220 乱数再起動）
- `src/pt.jl`
  - 49-64 lane / payload
  - 69-86 `_lane_segment!`
  - 92-96 `_swap_accepts`
  - 203-222 非同期レーン
  - 415-424 RNG の規律
- `src/observables.jl`
  - 85-98 `standard_observables`
  - 104-113 `_moment_sum`
  - 160-166 `standard_evaluables`
  - 207-248 `_finalize_stats`
- `src/checkpoint.jl`
  - 12 スキーマ版
  - 76-114 `_fingerprint`
  - 165-192 chain ブロック
  - 257-303 ヘッダと plan
  - 458-536 `resume`
- `src/hamiltonian.jl`
  - 234-266 `TiledHamiltonian`
  - 294-304 相異なるサイトのアサーション
  - 377-383 active / l1
  - 456-498 `_color_sites`
  - 549 `site_atom`
- `src/minimize.jl`
  - 54 `_gradient!`（既存 `zrows` を受け取る）
  - 96 `_minimize!`
- `test/unit/test_pt.jl:169-183`（payload の完全分割テスト）
- `test/unit/test_checkpoint.jl:21-27`（`_poison_pair`）
- `test/unit/fixtures.jl:94-131`（手組みの 2 / 3 / 4 体鎖。U1 のゲートに使える）
- 決定記録: `docs/specs/` 配下
  - `updates-stationarity.md` U1–U5
  - `pt-threads-determinism.md` P1–P6
  - `checkpoint-schema.md` C1–C4
  - `binning-observables.md` B3
  - `zeeman-field.md` Z1–Z5
