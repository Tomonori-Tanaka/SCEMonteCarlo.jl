# Design: constrained Monte Carlo (CMC) for finite-temperature anisotropy

Status: agreed (2026-11-21; draft v2.1) — spec-review（major 11 / minor 17）と numerical-review（major 4 / minor 11）、および v2 の再確認（各 major 1）を反映した。
v1 からの主な変更点：
- pt.jl を変えない mover の通し方にした。
- checkpoint の書き込み側を設計した。
- トルクの規格化を一意にした。
- 項の部分集合ごとのトルク（ユーザー決定）を入れた。
- `:auto` の弱い拘束に警告を出す（ユーザー決定）。
- 負の重みのフィクスチャと、N=3 の線形応答ゲートを追加した。
- 入口・初期化・アロケーションのゲートを追加した。

## Summary

主手法は Asselin 型の CMC で、重み付き M（Toga 型）と、重み 0 のサイトを通常の Metropolis で動かすハイブリッド更新を足す。

1 回の CMC 試行は次の順で進む。
1. 拘束サイト i について、補償スピン j を状態によらない一様な規則で選ぶ。
2. i を対称な回転提案で動かす。
3. j の ⊥ 成分を重み付きで補償する。
4. j の ∥ 成分は、符号を保って単位長に戻す。
5. 幾何的に不可能なら null move とする。
6. 可能なら、2 スピン同時のエネルギー差を逐次 leave-one-out で正確に計算し、ヤコビアン付きの Metropolis で判定する。

1 スイープは、活性サイトをサイト番号順に 1 回ずつ試す。拘束サイトはペア試行、重み 0 の活性サイトは 1 スピン Metropolis で試す。スイープは直列で回す。並列化は拘束方向（θ 点）ごとの独立レーンだけで行う。

観測量は次の 2 つ。
- 全トルク T = −Σ e × ∂E/∂e（公開の `energy_gradient!` から作る）
- 項の部分集合ごとのトルク（部分 Hamiltonian から作る。on-site と二イオンの分離用）

`ChainState`、`pt.jl`、fingerprint、GPU、既存のスイープには手を入れない。checkpoint には kind `"cmc"` を追加扱いで足す。

## 0. 制約チェックリスト（既存の約束 / 出典 / この設計での履行）

| # | 既存の約束 | 出典 | この設計での履行 |
|---|---|---|---|
| C1 | `(4π)^(body/2)` は ctor で 1 回だけ、`j0` は除外 | CLAUDE.md Numerical; `hamiltonian-tiling.md` | CMC は `H.terms` と既存カーネルだけを使う。部分 Hamiltonian も公開の ctor で作るので、スケールは 1 回だけ（§2.3, §4.3） |
| C2 | ΔE 局所性（インスタンス内のメンバーサイトは相異なる） | CLAUDE.md; `hamiltonian.jl:294-304` | 2 スピンの ΔE は単一サイトの leave-one-out を 2 回使う。新しい前提は i ≠ j だけで、パートナー抽選の形で保証する（§2.3） |
| C3 | `temperature` xor `kT`、β は受理判定だけ | CLAUDE.md | `resolve_kt` を流用。β は log 受理比の中だけに入る（§2.2） |
| C4 | U1：色クラス並列、serial ≡ parallel をビット一致、**サイト選択に RNG を使わない**、色順の走査 | `updates-stationarity.md` U1; `test_parallel.jl` | CMC は U1 に乗らない。直列だけで、`sweep_tasks` は無い。走査はサイト番号順で、パートナー選択に RNG を使う。どちらも CMC の規約として `constrained-mc.md` で宣言する。既存スイープは変えない（§2.5, §3） |
| C5 | U2：一様乱数は必要なときだけ引く | `updates-stationarity.md` U2 | ℓ < 0 のときだけ受理用の乱数を引く。null move では引かない（§2.2） |
| C6 | U3：適応は熱化中だけ、測定中は凍結 | `updates-stationarity.md` U3 | `acc_metro` / `att_metro` と `_adapt_step!` を流用する。ステップが下限に張り付いたら警告する（§2.6） |
| C7 | PT 決定論、`_swap_payload!` の全フィールド分割 | `pt-threads-determinism.md`; `test_pt.jl:169-183` | `pt.jl` は変えない。既存の `_compound_sweep!(st, H, β, scs, plan)` は残し、CMC 用は別メソッドにする（§1）。`ChainState` にフィールドを足さない（§2.5） |
| C8 | checkpoint schema v3、resume ≡ 中断なし（測定途中で中断）、温度境界の無条件書き込み | `checkpoint-schema.md`; CLAUDE.md; `run.jl:228-229` | kind `"cmc"` を追加扱いで足す。書き込み・読み込み・早期 return の分岐は §5。G13 は `_poison_pair` で測定途中に中断し、位置を assert する |
| C9 | checkpoint-schema の C3（温度境界の書き込みは、復元した RNG で `carryover=false` の再起動抽選をやり直す） | `checkpoint-schema.md` C3; `run.jl:217-218` | CMC の再初期化は RNG を消費しない決定論的な配置にする。schema 文書には kind ごとの記述として追記する（§2.8, §5） |
| C10 | `_fingerprint` は依存パッケージに固定されている | CLAUDE.md; `zeeman-field.md` | 変えない。拘束は `constraint/` group に保存する（§5） |
| C11 | 非活性サイトの規約（飛ばす・観測から除外・ビット凍結） | CLAUDE.md; `binning-observables.md` B3; `updates-stationarity.md` U1 | M・移動元・パートナー・観測量のすべてから除外し、凍結のまま。初期化も RNG を使わない決定論的な値にする（§2.1, §2.8、ユーザー決定） |
| C12 | Zeeman は fitted 項の後ろに付く body-1 項、magmoms ≥ 0 | `zeeman-field.md`; `hamiltonian.jl:414` | ΔE とトルクには既存カーネル経由で入る。トルクの分離では Zeeman 分を fitted の on-site と分ける（§4.3）。`:auto` の重みは常に非負になる（§4.1） |
| C13 | 観測量の規約は `binning-observables.md` の 1 か所 | CLAUDE.md | `:torque`、部分トルク、`:M_par`、`:constraint_residual` の定義と規格化を B 項として追記する（§4.2） |
| C14 | 勾配カーネルの連動 | CLAUDE.md | 公開の `energy_gradient!` を呼ぶだけで、カーネルには触れない（§4.2） |
| C15 | `metropolis_sweep!` の `att_metro += n_active`、`TempResult` の acceptance の意味 | `updates.jl:168`; `run.jl:53-62` | CMC は試行を 1 回ずつ数える。`CMCResult` の docstring で acceptance の意味（null を含む全試行の受理率）を定義する（§2.6, §4.1） |
| C16 | 結果の表示（`_print_points_table`、`Base.show`） | `run.jl:90-116` | `CMCResult` に専用の `show` を作る。χ・U の列は出さず、トルクの列を出す（§4.1） |
| C17 | 再現性の範囲 P6 | `pt-threads-determinism.md` P6 | `run_cmc` も同じ範囲（同じパッケージ版と Julia 版、スレッド数によらない）。`constrained-mc.md` に明記する |
| C18 | CPU ≡ GPU | `gpu-prototype.md` | `src/gpu/` に触れない |
| C19 | 依存パッケージ向けの名前 | CLAUDE.md | 変えない。追加だけ |
| C20 | 1 スイープあたりアロケーション 0、ホットパスの前後計測 | CLAUDE.md Performance | `cmc_sweep!` を `bench_sweeps.jl` に入れ、allocs = 0 をゲートにする（G16）。M5 のリファクタの前後で `bench_run.jl` / `bench_sweeps.jl` を計測し、`.claude/bench_log.md` に記録する |
| C21 | オラクルは実装から独立、σ は余裕つき | パッケージ横断の Testing 規約 | §7 の各ゲートにオラクルと、分解すべき変異を明記する |
| C22 | 公開 API・物理規約・詳細つり合い・スキーマ・層をまたぐリファクタはユーザー確認 | CLAUDE.md "Always confirm" / "Propose before" | §8 にまとめ、この仕様の合意をもって確認とする |

## 1. Module layout

| Target | Change |
|---|---|
| `src/cmc.jl`（新規。`run.jl` の後、`checkpoint.jl` の前に include） | `MagnetizationConstraint`、内部の解決済み型 `_ResolvedConstraint`、`_CMCScratch`、`_pair_delta_energy!`、`_cmc_pair_proposal`、`_attempt_cmc_pair!`、`cmc_sweep!`（内部）、拘束付き初期化、mover（§3.1）、`run_cmc`（単一とレーン）、`CMCResult`（+ `show`）、`cmc_observables`、`cmc_evaluables`、`torque_observable` |
| `src/observables.jl` | `total_torque`（public、未 export。型 `MagnetizationConstraint` に依存しないので、ここに置いても include 順の問題は無い） |
| `src/run.jl` | `_compound_sweep!` / `_run_temperature!` / `_mc_loop!` に mover の引数を足した**新メソッド**。既存の引数のメソッドはそのまま残し、新メソッドに既定の mover を渡して委ねる。`carryover = false` の再初期化と結果型の構築を mover に委ねる。`pt.jl` が呼ぶ 5 引数の `_compound_sweep!` は、計算順もビットも変わらない |
| `src/checkpoint.jl` | `_Checkpointer` に `constraint::Union{Nothing,_ResolvedConstraint}` を足す。`_write_header` で `constraint/` を書く。`resume` で kind と `constraint/` を先に読み、観測量の既定値をそのあとで解決する。完了済みファイルの早期 return も kind で分岐する |
| `src/SCEMonteCarlo.jl` | include の追加。export：`run_cmc`, `CMCResult`, `MagnetizationConstraint`, `cmc_observables`, `cmc_evaluables`, `torque_observable`。public：`total_torque` |
| `test/unit/test_cmc_*.jl`（新規、複数）、`test/runtests.jl` | §7 のゲート。`_poison_pair` は `test/unit/fixtures.jl` に移す |
| `bench/bench_sweeps.jl`、`bench/validate_cmc_*.jl`（新規） | CMC スイープの allocs と時間。手動検証の G8、G9、G11 |
| `docs/specs/constrained-mc.md`（新規、英語） | 決定記録。拘束測度、受理則、null move、パートナー規則、スイープの定義と RNG 消費、レーンの seed、トルクの規約、変異試験の記録 |
| `docs/specs/binning-observables.md` / `checkpoint-schema.md` / `pt-threads-determinism.md`（P6 に 1 行） / `docs/specs/README.md`（決定記録の表と仕様フォルダの表） | 追記 |
| `docs/src/guide/constrained_mc.md`、`docs/src/theory/`、`docs/src/api.md`、`docs/make.jl`、`SPEC.md`、`CHANGELOG.md`、`README.md` | 利用者向け |
| `CLAUDE.md` | 「Coupled code sites」に CMC の項目、「Standing contracts」に `constrained-mc.md` を足す |
| `.claude/agents/*.md` | test-runner の表、numerical-reviewer の決定記録一覧、code / maintainability / performance / spec-reviewer の層とホットパスの記述 |

`updates.jl` には入れない。単一サイト更新の契約の正本なので、CMC は別ファイルにする。

## 2. アルゴリズム（規約を含む）

### 2.1 拘束アンサンブル

- 拘束サイトの集合 𝒞 = { s : `site_active(s)` かつ w_{a(s)} ≠ 0 }。
- M = Σ_{s∈𝒞} w_{a(s)} e_s（符号付き重み）。M_∥ = M·n̂、M_⊥ = M − M_∥ n̂。
- 目標の測度（有限 N で厳密に定義。research/A §1.2、C §1.1）：

  π(dξ) ∝ ∏_s dΩ_s · e^{−βE(ξ)} · δ_{S²}(M̂(ξ), n̂),  δ_{S²}(M̂, n̂) = |M|² δ²(M_⊥) Θ(M_∥)

  このとき F(n̂) = −kT ln P(n̂) + 定数 となる。P(n̂) はカノニカル集団での M̂ の確率密度（立体角あたり）で、Asselin の F と同じものである。
- **ゲージ写像**（numerical review で確認）：ẽ_s = sign(w_s) e_s とすると、符号付きの問題は |w| の問題に厳密に写る。補償則、符号の保存、ヤコビアン、回転提案の対称性がすべてこの写像と可換だからである。したがって負の重みに特別な扱いは要らない。

### 2.2 1 回のペア試行（i ∈ 𝒞）

1. **パートナー j**：𝒞 \ {i} から一様に選ぶ。`site_rngs[i]` から `rand(rng, 1:|𝒞|-1)` を 1 回引き、i の位置以降は 1 つずらす。棄却ループは作らない。
2. **主スピンの提案**：e_i′ = `_rotate(e_i, _random_unit(rng), step·randn(rng))`。軸を先に、角度を後に引く（既存の `_attempt_metro!` と同じ）。CMC では反転提案を使わない。
3. **補償**（座標系に依らない形。n̂ ≠ ẑ でもこのまま使う）：

   e_j⊥′ = e_j⊥ − (w_i/w_j)(e_i⊥′ − e_i⊥)、s2 = |e_j⊥′|²

   `!(s2 < 1.0)` なら **null move**（等号と NaN を含める）。

   e_j∥′ = σ_j √(1 − s2)。σ_j = (e_j∥ ≥ 0 ? +1 : −1) とし、e_j∥ = 0 のときは +1 と決めておく。e_j′ = e_j⊥′ + e_j∥′ n̂。
4. **M_∥′** = M_∥ + w_i(e_i′·n̂ − e_i·n̂) + w_j(e_j′·n̂ − e_j·n̂)。`!(M_∥′ > 0.0)` なら **null move**。
5. **ΔE**：§2.3。
6. **log 受理比**：ℓ = −βΔE + 2 ln(M_∥′/M_∥) + ln|e_j∥| − ln|e_j∥′|。
   ℓ ≥ 0 なら受理。そうでなければ `site_rngs[i]` から u を引き、ln u < ℓ なら受理。
   debug ビルドでは `isfinite(ℓ)` を assert する（ℓ = −Inf は ln|e_j∥| = −Inf の場合だけ正当。§2.8）。
7. **受理したら**：config[i]、config[j]、zrows の列 i と j を書き、M を増分で更新し、`dE[i] = ΔE` とする。
   **棄却したら**：何も書かない（列 i の仮書き込みは §2.4 の関数の中で元に戻っている）。

**null move も試行として数える**（`att_metro += 1`、受理用の乱数は引かない）。数えずにやり直すと、定常分布が π(x)(1 − r(x)) に偏る。
research/A §1.3 と numerical review の N=3 の検算で、それぞれ 8σ と 13σ のずれを再現した。

### 2.3 2 スピン同時のエネルギー差（逐次 leave-one-out）

ΔE = [E(e_i′, e_j) − E(e_i, e_j)] + [E(e_i′, e_j′) − E(e_i′, e_j)]

1. `fill!(lc, 0)`、`site_coeffs!(lc, H, i, zrows)` → ΔE_i = `delta_energy(lc, zrows[:,i], Z(e_i′))`
2. 列 i を退避し、Z(e_i′) を仮に書く。
3. `fill!(lc, 0)`、`site_coeffs!(lc, H, j, zrows)` → ΔE_j = `delta_energy(lc, zrows[:,j], Z(e_j′))`
4. 列 i を退避コピーからビット単位で戻す。

（拘束との混同を避けるため、leave-one-out 係数を `lc` と書く。）括弧内はどちらも単一サイトの移動なので、C2 のもとで何体項でも厳密である。
i, j がインスタンスを共有しなければ、結果は素朴な和とビット単位で一致する。幾何的な判定（§2.2 の 3–4）はエネルギー計算より前に行うので、null move では `site_coeffs!` を呼ばない。

### 2.4 内部 API（テスト用に切り出す）

`_cmc_pair_proposal(config, zrows, H, rc::_ResolvedConstraint, M, i, j, e_i′, β, sc) -> (e_j′, ΔE, ℓ)`

- 状態を書き換えない。zrows の列 i は関数を抜ける前に必ず元に戻す。
- null move のときは `(e_j, NaN, -Inf)` を返す（ΔE を誤って使ったら、それと分かるようにする）。
- `_attempt_cmc_pair!` はこの関数と、乱数と、受理時の書き込みだけで組む。G4 は ℓ をテスト側の独立な式と比べる。

### 2.5 スイープの定義、RNG 消費、状態

- **1 CMC スイープ**：`dE` を `fill!(0)` し、活性サイトをサイト番号順に回す。
  - s ∈ 𝒞：s を移動元とするペア試行を 1 回。
  - 活性で w = 0：既存の `_attempt_metro!` を 1 回（反転提案を含む）。これらのサイトは M に寄与しないので、拘束とは無関係に動かせる（ハイブリッド CMC）。
  - 最後に `_reduce_dE`（固定順の和）を `st.energy` に足す。
  - 固定順に合成した各カーネルが π を保つので、合成も π を保つ（U1 と同じ論法）。
- **RNG**：サイト s の試行は `site_rngs[s]` だけから引く。順番はパートナー添字 → 回転軸 → 回転角 →（必要なら）受理用の一様乱数。チェーンの `st.rng` はスイープ内では使わない。新しい RNG 状態は無い。
- **M**：各スイープの開始時に、config から決まった順序でゼロから計算する。スイープ内では増分で更新する。スイープをまたいで持ち越さないので、`ChainState` は変えない。
  開始時の再計算で `|M_⊥|/M_∥ > 1e-10` なら `@warn maxlog=1`（§2.6）。
- **`_CMCScratch`**：`lc`、`zi_new`、`zi_old`、`zj_new`、`plm` と、𝒞 の索引の配列をレーンごとに 1 回だけ確保する。スイープ中に確保は無い（G16）。

### 2.6 適応ステップ、統計、ずれ

- **カウンタ**：`acc_metro` / `att_metro` を流用し、ペア試行とハイブリッドの 1 スピン試行を合算する。`_adapt_step!` は変えない。
  CMC での `acceptance_metropolis` は「null move を含む全試行の受理率」と定義する。
- **ステップの下限**：熱化の終わりに step が clamp の下限（`updates.jl` の 1e-3）に張り付いていたら `@warn maxlog=1` を出す。
  原因は null が多いこと（重み比の大きいペア）だと示唆する。null を除いた率で適応させる案は O8 に残す。
- **再正規化**：`_renormalize!` はそのまま使う。M_⊥ の丸めのずれは監視するだけで、再射影はしない。
  補償則が M_⊥ を参照しないので、ずれはランダムウォークのまま残る。見積もりは 1e10 試行で 1e-11 rad で、トルクへの影響は無視できる（numerical review の問 6）。
- **弱い拘束の警告**（ユーザー決定）：熱化の終わりに `M_∥/Σ_{s∈𝒞}|w_s| < 0.05` なら `@warn maxlog=1` を出す。
  文面には原因を 2 つ挙げる。
  - Tc より上（有限 N の残りの M。常磁性では M_∥/Σ|w| ~ 1/√|𝒞|）。
  - 反強磁性・補償点付近（正味の M の向きの拘束は意味が薄い。反平行の副格子に負の重みを付けるか、副格子を w = 0 のハイブリッドにすること）。
- **初期配置**：`:auto` で `init` が無いと、強磁性の配置から始まる。反平行の副格子をもつ模型（Nd₂Fe₁₄B の l02 など）でも測度は正しいが、熱化に時間がかかる。guide では `find_ground_state` の結果を `init` に渡すよう勧める。

### 2.7 パートナー規則

- 𝒞 \ {i} から一様に選ぶ。選ぶ確率が状態によらないので、詳細つり合いは成り立つ（逆向きの手は同じ順序対 (i, j) で元の e_j に戻る）。
- **同じ重みクラスの中だけに限る規則は採らない。** クラスごとの ⊥ 和 Σ_{s∈A} w_s e_s⊥ が保存量になり、エルゴード性が壊れる（numerical review で確認）。
- 一様な規則は、次の重みで iid のオラクルと約 2σ 以内で一致した：(1, 0.4, 1.5)、(1, 1, 3)、Fe/Pt 型の比 8 (8, 1, 8, 1)、Néel (1, 1, −1, −1)。
  重み比が大きいペアは幾何的にほぼ棄却されるが、正しさは保たれる。効率の改善は O1 に回す。

### 2.8 拘束付きの初期化

- **既定のパターン**：s ∈ 𝒞 なら e_s = sign(w_{a(s)}) n̂。活性で w = 0 なら n̂。非活性サイトも n̂（凍結され、どの量にも入らないので、値は何でもよい。RNG を使わない）。
- **`init` を与えた場合**：その配置の M（𝒞 上）を計算し、全サイトを最小回転 R で M̂ → n̂ に剛体回転する。
  M̂ = −n̂ のときは、軸を `normalize(n̂ × x̂)` とする（n̂ ∥ x̂ なら ŷ）。
- **`carryover = false`**：各温度の開始で、同じ初期配置（既定のパターン、または回転済みの `init`）に戻す。RNG は消費しない。
- **入口で拒否する場合**：
  - |M_init| < 1e-8·Σ|w|
  - |𝒞| < 2
  - `weights = :magmoms` なのに `H.magmoms === nothing`
  - 重みベクトルの長さが cell atom 数と違う、または非有限
  - 方向がゼロまたは非有限
- e_s·n̂ が厳密に 0 のスピンを補償に選ぶと ln|e_j∥| = −Inf になり、その手は棄却される。測度は正しく、そのスピンは移動元としては動けるので、鎖は詰まらない。docs に書いておく。

## 3. 決定論と並列性

### 3.1 mover

内部の型 `_SweepMover`（既存の動作。単一サイト Metropolis と OR）と `_CMCMover`（rc と `_CMCScratch` を持つ）を作り、新しいメソッドが次を mover に委ねる。
- 1 スイープ
- `carryover = false` の再初期化
- 結果型の構築（`MCResult` / `CMCResult`）

checkpoint の `constraint/` group は mover ではなく `_write_header` が書く。元にするのは `_Checkpointer.constraint` で、書き込みの責任はここに一本化する（§5）。

既存の引数のメソッドは `_SweepMover` を渡して新メソッドに委ねる。`pt.jl` の `_lane_segment!` は既存の 5 引数の `_compound_sweep!` を呼び続ける。
その経路の計算順と RNG 消費が変わらないことを、G15 で確かめる。

### 3.2 単一チェーンとレーン

- **単一チェーン**：直列だけ。`run_cmc` に `sweep_tasks` は無い。同じ seed ならスレッド数によらずビット一致する。
- **レーン**：`run_cmc(H, cs::AbstractVector{MagnetizationConstraint}; ntasks, seed, …)`。
  - マスター `Xoshiro(seed)` から `rand(master, UInt64)` を k = 1, 2, … の順に引き、レーン seed にする。
  - 各レーンは単一チェーンのドライバを独立に回す。`CMCResult.seed` とレーンの checkpoint にはレーン seed を入れるので、1 レーンだけを単独で再現できる。
  - 結果は `ntasks` によらずビット一致する（G14）。
- **色クラス並列の CMC は作らない。** 受理則が M_∥ を通じて全ペアを結ぶので、インスタンスを共有しないペアどうしも可換でない。

## 4. API

### 4.1 型とドライバ

```julia
struct MagnetizationConstraint          # inner ctor normalizes and validates
    direction::SVector{3,Float64}       # n̂ (unit; zero / non-finite rejected)
    weights::Union{Symbol,Vector{Float64}}  # :auto | :uniform | :magmoms | per-cell-atom signed
end
MagnetizationConstraint(direction::AbstractVector{<:Real}; weights = :auto)
MagnetizationConstraint(θ::Real, φ::Real; weights = :auto)  # n̂ = (sinθcosφ, sinθsinφ, cosθ)
Base.:(==), Base.isequal, Base.hash   # field-wise (Vector field)

run_cmc(H::TiledHamiltonian, con::MagnetizationConstraint;
        temperature = nothing, kT = nothing,
        sweeps_therm = 2_000, sweeps_measure = 10_000, measure_interval = 1,
        step = 0.6, adapt_target = 0.5, adapt_interval = 50,
        renorm_interval = 1_000, nbins = 32,
        observables::Vector{Observable} = cmc_observables(H, con),
        evaluables::Vector{Evaluable} = cmc_evaluables(),
        init = nothing, carryover = true, seed = rand(UInt64),
        checkpoint = nothing, checkpoint_interval = 0)::CMCResult

run_cmc(H::TiledHamiltonian, cons::AbstractVector{MagnetizationConstraint};
        observables = con -> cmc_observables(H, con),   # a FUNCTION of the constraint
        ntasks = Threads.nthreads(), seed = rand(UInt64),
        checkpoint = nothing,                            # lane k -> "<stem>.lane<k>.jld2"
        kwargs...)::Vector{CMCResult}
```

- レーン版の `observables` は拘束の関数として受け取る。`Vector{Observable}` を渡すと拒否する（n̂ に依存する観測量が全レーンで共有されて、黙って誤るのを防ぐ）。
- **`CMCResult(points::Vector{TempResult}, constraint::MagnetizationConstraint, final_config, seed)`**
  - `constraint` には解決後の重み Vector を入れる。
  - `show` は T、⟨E⟩、トルクの 3 成分、⟨M_∥⟩、acceptance の表を出す。
- **重みの解決**：
  - `:auto` は、`H.magmoms !== nothing` なら magmoms、そうでなければ全部 1。magmoms ≥ 0 が強制されているので、`:auto` の重みは常に非負である。符号付きは明示したベクトルのときだけ。
  - `:magmoms` は magmoms が無ければ拒否する。
  - ベクトルは H の cell atom の順。`reduce_cell` を使った場合は縮約後の原子順になる。
  - 重みは cell atom ごとなので、磁気セルが化学セルより大きい反強磁性体（例：基本セルで扱う G-AFM）の Néel ベクトルは表せない。サイトごとの重みは v2。
- **ホットパス**には `_ResolvedConstraint(n̂::SVector, cell_w::Vector{Float64}, site_w::Vector{Float64}, cset::Vector{Int})` を渡す（型を安定させる）。
  - `cell_w` は cell atom ごとの解決後の重みで、checkpoint と `CMCResult.constraint` の元になる。
  - `site_w` と `cset` はそこから導出する。
- **`cmc_evaluables()`** は比熱だけ。拘束アンサンブルでも Var(E)/kT² = ∂⟨E⟩/∂T が厳密に成り立つ（参照測度が温度によらないため）。χ と U は拘束下で意味が変わるので外す。

### 4.2 トルクの規約と観測量

- **`total_torque(H, config)::SVector{3,Float64}`** は全系の和を返す。
  - T = −Σ_{s 活性} e_s × ∂E/∂e_s = Σ_s G_s × e_s。G は `energy_gradient!`（接平面へ射影済み）で、Zeeman 込み。
  - 重みは掛けない。拘束が重み付きでも T は重みなしの和である。
- **恒等式**（全系の F について、有限 N で厳密）：⟨T⟩ = −n̂ × ∂F/∂n̂。
  - 成分：⟨T⟩·ê_φ = −∂F/∂θ、⟨T⟩·ê_θ = (1/sinθ)∂F/∂φ、⟨T⟩·n̂ = 0。
  - ê_φ = (−sinφ, cosφ, 0)、ê_θ = (cosθcosφ, cosθsinφ, −sinθ)。
  - 符号の検算：E = −K Σ(e·ẑ)²、e = (sinθ, 0, cosθ) で T·ŷ = −K sin2θ。E = −K(e·x̂)²、θ = π/2 で T·ê_θ = K sin2φ（numerical review）。
- **観測量 `:torque`** = `total_torque / n_cells`（訓練セルあたり、3 成分の Cartesian）。これは F/n_cells の恒等式に対応する。
  ê_θ と ê_φ は θ = 0, π で定義できないので、極座標への射影は guide 側（後処理）で行う。
- **`cmc_observables(H, con)`**：
  - `:energy`、`:energy2`（既存と同じ、全系）
  - `:torque`（3 成分、セルあたり）
  - `:M_par` = M·n̂ / n_cells（重みの単位）
  - `:constraint_residual` = |M_⊥|/M_∥（ラジアン）
  - H に磁場があれば、`:torque_zeeman` = MU_B_EV_T·(M_μ × B)/n_cells も加える（解析式。M_μ = Σ m e は magmoms の重み）。
  - 観測量関数は純関数で、可変スクラッチを捕捉しない。

### 4.3 項の部分集合ごとのトルク（ユーザー決定：v1 に入れる）

- **`torque_observable(name::Symbol, Hpart::TiledHamiltonian, H::TiledHamiltonian)::Observable`**
  - 部分 Hamiltonian のトルク `total_torque(Hpart, config)/n_cells` を測る観測量を返す。
  - 入口で次を検査する。
    - `Hpart` と `H` で `n_cell_atoms`、`dims`、`n_sites` が一致する。
    - `Hpart` が磁場を持たない。
    - `Hpart` の各 ScaledTerm（atoms、shifts、ls、folded、coef）が、H の fitted 項 `H.terms[1:n_fitted_terms]` のどれかと一致する。
    - 原子数が偶然同じでも、別の項リスト（例えば `reduce_cell` の前の模型）から作った `Hpart` を通さないためである。
  - `Hpart` は不変オブジェクトなので、レーン間で共有して読んでも安全である。
- **作り方**（guide に書く手順）：`Hpart = TiledHamiltonian(n, filter(pred, terms); dims)`。磁場は付けない。
  - `terms` は H を作ったのと同じ項リストにする（`reduce_cell` を使ったなら `red.terms`）。
  - 分割は互いに素で、fitted 項を全部覆うようにする。そうでないと部分トルクの和は全トルクにならない（関数側では強制しない）。
  - 部分集合が空だと（例えば body ≥ 3 の項が無い場合）、ctor は `ArgumentError` を出す（`hamiltonian.jl:279-283`）。guide に書く。
  - 例：on-site = `t.body == 1`、二イオン = `t.body == 2`、高次 = `t.body ≥ 3`。
  - 公開の ctor を通るので、スケールは 1 回だけ（C1）。
  - Zeeman 分は `:torque_zeeman` が受け持つ。
- **恒等式**：fitted 項を互いに素に分ければ、配置ごとに Σ_parts T_part + T_Zeeman = T_total が成り立つ（G17）。
  - ただし個々の ⟨T_part⟩ は、O(K) を超えると F の勾配場とは限らない（∇⟨H_c⟩ との差は −β Cov の項。research/A §6 O9）。
  - on-site と二イオンへの分け方が一意になるのは O(K) の極限だけである。この解釈上の注意を決定記録と guide に書く。
- **後処理**（K の当てはめ、F の積分、κ_lm のフィット）は対象外。guide で最小二乗の例を示すだけにする。

## 5. Checkpoint

- kind `"cmc"`（schema v3 のまま、追加扱い。`gpu_pt` の前例）。
- **書き込み**
  - `_Checkpointer` に `constraint` を足す。
  - `_write_header` が `constraint/{direction(3), weights(n_cell_atoms, 解決後), partner_rule = "uniform"}` を書く。
  - `plan/or_per_metropolis = 0` と `plan/sweep_tasks = 1` を固定値として書く。
  - チェーンのブロックは既存の `_write_chain` をそのまま使う。
  - 温度境界での無条件の書き込みも、既存と同じにする。
- **読み込み（`resume`）**
  - シグネチャを `resume(path, H; observables = nothing, evaluables = nothing, …)` にする。
  - 順序は次のとおり。
    1. ファイルを開いて kind を読む。`"cmc"` なら `constraint/` も読む。`partner_rule ≠ "uniform"` ならエラー。
    2. 既定値を解決する。`"mc"` / `"pt"` は `standard_observables(H)` / `standard_evaluables()`、`"cmc"` は `cmc_observables(H, con)` / `cmc_evaluables()`。
    3. `_check_observables`、名前の照合、`_check_evaluables` を、今と同じ内容のまま実行する。
  - 完了済みファイルの早期 return も kind で分岐する。
  - 戻り値の Union に `CMCResult` が加わる。
  - 明示的に渡した値の扱いと、既存 kind の結果は変わらない。
- **古い版で開いた場合**：古いパッケージで `"cmc"` ファイルを既定の引数のまま開くと、"unknown kind" に届く前に観測量名の不一致エラーで落ちる。checkpoint-schema.md にはこう正確に書く（`gpu_pt` の前例とは失敗の仕方が違う）。
- **レーン**は 1 レーン 1 ファイル。ファイル名は `<stem>.lane<k>.jld2` で、`checkpoint` が `.jld2` で終われば stem はその拡張子を除いたもの、そうでなければ与えたパスそのもの。レーン全体をまとめた resume は O6。

## 6. Impact on coupled sites

- [ ] `hamiltonian.jl` ↔ SCEFitting introspection contract — 影響なし
- [ ] `energy.jl` 4-function contract ↔ `updates.jl` — 呼ぶだけ。G2 を新設する
- [ ] `lm_index` ↔ `zlm_row!` ↔ OR axis — 影響なし（CMC は OR を使わない）
- [ ] `reduce.jl` ↔ tiling ↔ `geometry.jl` ordering — 影響なし。重みの原子順だけ docstring に書く
- [ ] Gradient kernels — 呼ぶだけ（`total_torque` は公開の `energy_gradient!` 経由）
- [x] Checkpoint writer ↔ reader ↔ `checkpoint-schema.md` — kind `"cmc"`、`_Checkpointer`、`_write_header`、`resume` の解決順、早期 return
- [ ] Coloring ↔ sweeps ↔ `updates-stationarity.md` — 影響なし。CMC の定常性は `constrained-mc.md` に書く
- [ ] Device row / kernel / gradient — 影響なし
- [ ] CPU PT ↔ GPU PT — 影響なし（5 引数の `_compound_sweep!` を残す）
- [x] Inactive-site convention — CMC 側で守る（G1b）
- [x] `.claude/agents/` references — §1 の表のとおり
- [x] `SPEC.md` / `docs/src/api.md` / guide pages — 追加する
- [x] `run.jl` 内部ドライバ — mover の新メソッド（§3.1、G15）
- [x] `CLAUDE.md` — Coupled code sites と Standing contracts に追記する

## 7. Test strategy

### 7.1 フィクスチャ

すべて手書きの `MultipoleTerm` で作る（`fixtures.jl` の流儀）。J と K はエネルギー差から読み取り、式を推測しない。
拘束方向は、特に断らない限り一般の n̂（θ = 0.6、φ = 0.35）にする。θ = 0 では、実験室座標系で補償する誤りが見えない。

| 名前 | 内容 |
|---|---|
| F2u / F2g | 2 原子、dims = (1,1,1)。等方 l = 1 ペア ＋ 各原子に body-1 l = 2 項。F2u は m = 0 だけ（一軸）、F2g は m = −2..2 に乱数（φ 依存） |
| F2w± | F2g に明示の重み (1, 2) と (1, −2) を付けたもの |
| F3 | 3 原子。全対ペア ＋ 3 体項 ls = [1,1,1]、folded[2,2,2]（時間反転について奇）＋ 単イオン l = 2 |
| F3a | F3 に、異方的な二イオン項（l=1⊗l=1 で folded が単位行列でないもの、と l=2 を含むペア）を足したもの。単イオン項の folded は原子ごとに乱数で変える（G17 用。等方ペアのトルクは配置ごとに恒等的に 0 なので、二イオン部分のずれが見えない） |
| F3iso | 3 原子の等方な環（`_chain_terms`、dims = (3,1,1)）。E = J(R² − 3)/2 |
| F4 | 4 原子。4 体 folded[2,2,2,2]（`_fourbody_terms` 型） |
| Fmix | 8〜27 サイト。符号の混じった不等な重み (+1.0, −0.4, +1.5, …)。1 サイトは重み非零の非活性サイト、1 サイトは重み 0 の活性サイト |
| FZ | F2g に magmoms と磁場を付けたもの |
| FL | sc 4×4×4 の等方強磁性 ＋ 微小な単イオン K（手動検証用） |

E = 0 のアンサンブルは、body-1 の l = 0 項で全サイトを活性にできれば求積なしで使える。ctor が受け付けるかは M1 で確かめる。受け付けなければ、等方結合の求積で代える。

### 7.2 ゲート

| ゲート | 種類 | オラクル | 殺す変異（見えないフィクスチャ） |
|---|---|---|---|
| G0 入口・初期化・警告 | 決定論 | §2.8 の拒否の全項目。既定のパターンと、回転した `init` で M̂ = n̂（1e-14）。M̂ = −n̂ の `init`。`carryover = false` の再初期化で RNG の状態が変わらない（`copy` と `==`）。3 つの警告（弱い拘束、ステップの下限張り付き、\|M_⊥\|/M_∥ > 1e-10）が条件を満たすときに出て、満たさないときには出ないことを `@test_logs` で確かめる。拘束のずれは、テストで M_⊥ を人工的に入れて起こす | 入口の漏れ、sign(w) の欠落（Fmix で見える）、再初期化の RNG 消費、警告の条件の誤り |
| G2 ペア ΔE | 決定論、ms | `total_energy` の差（1e-12、最悪 < 1e-13）。2/3/4 体の同じインスタンス（4 体の端の対 (1,4) を含む）、複数のインスタンスを共有、共有しない。`inst_ptr` で共有を assert する。棄却後の zrows はビット単位で元に戻る。共有しない対は素朴な和とビット一致 | 素朴な ΔE_i + ΔE_j、更新順の誤り、巻き戻し漏れ（共有しない対だけでは見えない） |
| G1 拘束の不変量 | 決定論、ms | テスト側でゼロから計算した M_⊥。10⁴ 回の移動後と `_renormalize!` の直後に ≤ 1e-12·Σ\|w\|。M_∥ > 0、\|e\| = 1、j ≠ i。Fmix と F2w± を使う | 実験室座標系での補償（θ = 0 では見えない）、重みの取り違え、補償比を \|w_i/w_j\| にする誤り（正の重みだけでは見えない）、j = i |
| G1b 非活性・重み 0 | 決定論 | Fmix で、非活性サイトがビット凍結され M に入らない。重み 0 の活性サイトは動くが M は不変 | 非活性サイトを 𝒞 に入れる誤り（非活性サイトの重みが 0 なら見えない） |
| G3 トルクの一点恒等式 | 決定論、ms | `total_energy` を大域回転した中心差分で T·w を求める（w ∈ {ê_θ, ê_φ, n̂}）。等方模型では \|ΣT\| < 1e-12。FZ では T_Z = μ_B M_μ × B を手計算 | 符号、軸、因子、Zeeman の漏れ |
| G17 部分トルクの和 | 決定論、ms | F3a と FZ で、body 次数で分けた部分 H のトルクの和 ＋ T_Z = 全トルク（1e-12）。各部分も G3 と同じ中心差分（その部分の `total_energy`）と一致する。入口の検査では、別の項リスト・磁場つき・dims 違いの `Hpart` を拒否する | 部分 H のサイト番号のずれ・原子の並べ替え（等方ペアや原子ごとに同じ folded では見えない）、スケールの二重適用、Zeeman の二重計上 |
| G4 受理比の恒等式 | 決定論、ms | `_cmc_pair_proposal` の ℓ を、テストで独立に書いた −βΔE_total + 2 ln(M′/M) + ln\|z_j\| − ln\|z_j′\| と比べる（ΔE は `total_energy` の差）。逆提案で元に戻る（e_j∥ < 0 の状態を含む）。不正な状態で −Inf、ΔE は NaN | ヤコビアンの指数・\|z\| 比・z 符号の分岐・M_∥′ ≤ 0・β の二重適用。注：ℓ(x→y) = −ℓ(y→x) は ℓ = f(y) − f(x) の形なら何でも通るので、指数の誤りは殺せない。殺すのは独立な式との比較と G5/G6 |
| G5 幾何ゲート | 統計、< 1 s | 定理 (I)：等方なら、回転不変量の分布は拘束しないときと同じ。N=2 で ⟨M_z⟩ = 4/3。N=3 で 13/8、P(e·n̂ < 0) = 1/8。重み付き (1, 2) と (1, −2)（ゲージ写像で同じ値）で 13/6。結合がある場合は e^{−βE(R)} で重み付けした 1–2 次元の求積。**step を固定して適応を切り**、その値を記録する。atol = 5σ | ヤコビアン欠落、指数、\|z\| 比、null を数えないやり直し、z 符号の固定。N=3 が要る（等重みの N=2 では「指数 1 かつ \|z\| 欠落」と z 符号が見えない） |
| G6 N=2 トルク vs 求積 | 統計、1–2 s | 決定論的な求積 Z₂(n̂)（research/A §6 O4、C 式 (5)）。格子を倍にして < 1e-10 を assert。重み付きは小さいモーメントのスピンを積分変数にとる。F2u、F2g、F2w± で、3 成分 | パイプライン全体。二量体では必ずインスタンスを共有するので素朴な ΔE、θ/φ の規約、トルクに m を掛ける誤り、負の重みの扱い |
| G7 N=3 線形応答 | 統計、< 1 s | E_iso = 0、βK ≈ 1e-3 で K_eff = K·⟨P₂(e·M̂)⟩_iso = K·0.220782。3 原子・dims (1,1,1)・E_K = −KΣ(e·ẑ)² なら、セルあたり ⟨T⟩·ê_φ = −3·0.220782·K·sin2θ。値はテスト内で 2 次元 Gauss–Legendre 求積で出す。被積分関数が R=0 で不連続なので r=1 で区間を分け、自己収束の許容は 1e-6（G6 の 1e-10 は流用しない）。K を 2 通り試して O(βK) が 5σ 未満であることを示す | ヤコビアンの全変異（N=3 で 0.107〜0.330 と大きく動く）、規格化 |
| G13 resume | 決定論 | 測定途中で中断し（位置を assert）、中断なしの run と `==`。拘束の round-trip（`==` を定義済み）。既存 kind のファイルが従来どおり読める | n̂・重みの保存漏れ、kind の分岐漏れ |
| G14 レーン | 決定論 | `ntasks = 1` ≡ `ntasks = 4` で `==`。レーン k の結果が、レーン seed で単独に回した結果と `==`。`observables` に Vector を渡すと拒否 | seed の導出順、観測量の共有 |
| G15 既存ドライバの中立性 | regression pin（変化検出器と明記、正しさの根拠ではない） | リファクタ前のコミットで、手書きの項のフィクスチャを使い、`run_mc`（OR あり・なし）と `run_pt` の固定 seed 結果（観測量の平均と最終配置）を取る。rtol 1e-10。取得コミットと再取得の条件をコメントに書く。M5 の作業中には、リファクタの前後で同じ OS・同じ Julia のビット一致も一度確かめ、tasklist に記録する | mover リファクタによる既存経路の変化（乱数消費や更新順が変わると O(1) で動く） |
| G16 アロケーション | 決定論 | `cmc_sweep!` の 2 回目以降の呼び出しで `@allocated == 0` | スイープ内の確保 |
| G8 多体合成 | 手動（短くできればユニットの中枠） | 回転 iid 参照（research/C §1.4）。F3 と F4、時間反転について奇 | 多体の合成、M_∥′ ≤ 0 の漏れ（偶の模型では見えない） |
| G9 大きい N の線形応答 | 手動・夜間 | K_eff/K = ⟨P_l(e·M̂_w)⟩_iso。右辺は既存の `run_mc` で等方模型を回し、拘束と同じ重み・同じ活性サイトの集合で作った M̂_w で測る | 示量的なずれ（\|z\| 因子）。M_z² は見えない（G5/G6 の担当） |
| G11 ヒストグラム | 手動 | 拘束しない `run_mc` での M̂_w のヒストグラム。βΔF ≲ 4、N = 8–27 | 中間の N の相互作用格子 |
| G12 本番スモーク | 手動 | ⟨n̂·T⟩ = 0。時間反転について偶で B = 0 なら ∫₀^π τ dθ = 0。面内の 2 回軸か鏡映があれば τ(π/2) = 0 | 規格化、閉路 |

- **統計ゲートの書き方**：test_zeeman と同じ書式で、seed、測った σ（binning の err）、ずれ、atol = 5σ、分解すべき変異のシフトをコメントに残す。
  `err < ε_max` も assert して空振りを防ぐ。各変異は実装時に一時パッチで当てて落ちることを確認し、結果を決定記録に残す。
  事前に確かめた感度は research/C §2 と research/scripts/review/ にある（numpy の試作値なので、pin には使わない）。
- **「等方模型なら平均トルク 0」は統計ゲートにしない。** 配置ごとに ΣT = 0 が成り立つので、どの変異でも 0 になる。G3 の一点恒等式に格下げする。
- Callen–Callen の漸近則はゲートにしない（docs の物理例だけ）。

## 8. ユーザー確認が要る項目（この仕様の合意で確認とする）

1. **公開 API の追加**
   - export：`run_cmc`（2 メソッド）、`CMCResult`、`MagnetizationConstraint`、`cmc_observables`、`cmc_evaluables`、`torque_observable`
   - public：`total_torque`
   - `cmc_sweep!` と `_CMCScratch` は内部にとどめ、public にしない（`metropolis_sweep!` が public である前例とは分ける。依存パッケージからの需要が出てから決める）。
   - `torque_observable` は入口で、磁場つき・別の項リスト・dims 違いの `Hpart` を拒否する。
   - export はすべて M5 でまとめて入れ、`api.md` も同時に更新する。
2. **`resume`**：既定値を `nothing` にして、kind と拘束を読んでから解決する。戻り値に `CMCResult` が加わる。
3. **checkpoint**：kind `"cmc"`、`constraint/`、`plan` の固定値を、schema v3 のまま追加扱いにする。レーンのファイル名は `<stem>.lane<k>.jld2`。
4. **`_fingerprint`** は変えない。
5. **物理規約と観測量の定義**
   - 拘束測度（δ_{S²}、つまり M_∥² 因子込み）
   - トルクの符号
   - 規格化（`total_torque` は全系、`:torque` と部分トルクはセルあたり、重みなし）
   - θ と φ の規約
   - `:M_par`、`:constraint_residual`（ラジアン）、`:torque_zeeman`
   - CMC での acceptance の意味
   - 既定の evaluables から χ と U を外す
   - 重みの既定（`:auto`）と弱い拘束の警告
6. **詳細つり合いの規則**
   - Asselin の受理則
   - null move を数える
   - パートナーは 𝒞\{i} から一様
   - CMC では反転提案を使わない
   - 重み 0 のサイトはハイブリッドの 1 スピン Metropolis
7. **決定論**
   - CMC は直列だけ
   - 走査はサイト番号順で、パートナー選択に RNG を使う（U1 とは別の規約）
   - レーンの seed 導出
   - 再初期化は RNG を消費しない
8. **層をまたぐリファクタ**：`run.jl` の内部ドライバに mover の新メソッドを足す（5 引数の既存メソッドは残す。G15）。
9. **プロセス**
   - `CLAUDE.md` と `.claude/agents/` の編集
   - SLCEMonteCarlo.jl への移植はしない
   - `research/`（報告と試作スクリプト）をこの仕様フォルダとして公開リポジトリにコミットする

## 9. Risks and open items

- **O1 重み比が大きいときの効率。** FePt で Pt を独立スピンにすると、Fe を主・Pt を補償とする試行の多くが幾何的に棄却される。
  v1 は正しさを優先して一様に選ぶ。本番のモデルで null の割合を bench スクリプトで測り（結果型や checkpoint には入れない）、v1.1 で状態によらない偏った規則を検討する。
  例は「|w_j| ≥ |w_i|」に偏らせる規則や、ペアの種類ごとのステップ幅。同じクラスの中だけに限る規則は不可。
- **O2 再正規化のずれ**：監視だけにする（§2.6）。G1 の上限で固定する。
- **O3 エルゴード性**：ゲージ写像により、大きさの等しい ± 副格子は Asselin App. B に帰着する。未証明なのは |w| が等しくない場合だけで、G5/G6 の重み付き版で分布の一致を見て間接的に確かめる。
- **O4 有限 N での F の意味**：1 スピンあたり O(kT/N) の測度依存の部分がある。docs に書き、2 サイズでの外挿を推奨する。
- **O5 トルク測定のコスト**：`energy_gradient!` は呼ぶたびに zrows を確保して再構築する。部分トルクの数だけコストが増える。
  v1 は `measure_interval` で吸収し、必要になれば内部パスを足す。
- **O6 レーン全体の resume**：v1 は 1 レーン 1 ファイル。
- **O7 θ 間のウォームスタート**：レーン並列と両立しない。直列スキャンのヘルパは v1.1 の候補。
- **O8 カウンタの合算と null**：null を除いた受理率で適応させる案、ハイブリッドとペアで別のステップ幅を使う案。どちらも `ChainState` を変えずにできるかを、需要が出てから検討する。
- **O9 Asselin App. B の証明の穴**（Lemma 1 の最後の 1 手）：実用上は問題ないと判断する。
- **O10 Nd₂Fe₁₄B の重みの物理的な意味**：DFT のスピンモーメントの重み（`:auto`）は、4f の全モーメントの向きとは別物である。スピン再配向の温度域では、F_total と F_Fe の差が物理として効く。
  guide では「Fe だけを拘束し、Nd は w = 0 のハイブリッドにする」案と、Toga 型の全重みの案を並べて示す。
- **O11 サイトごとの重み**（磁気セル ≠ 化学セルの反強磁性体の Néel ベクトル）は v2。
