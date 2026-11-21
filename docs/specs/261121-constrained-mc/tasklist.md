# Tasklist: constrained Monte Carlo (CMC) for finite-temperature anisotropy

Status: landed (2026-11-21) — M0–M8 完了、Tier 2 パネル（ステップ A・B・C）と着地前の 2 本のレビュー（`773b7df..1ac5601`、blocker 0）の指摘を反映済み

このファイルには、コミット単位の大まかなマイルストーンを置く。日々の追跡はセッション内の `TaskCreate` で行う。

各マイルストーンでは、実装の前に「新しく宣言する保証」と「維持する保証」を挙げる。
新しい保証には、落とすべき誤りと、それが見えるフィクスチャを添える。
実装が終わるたびに、別エージェントのレビュー（numerical-reviewer。独立な変異試験を含む）を通してから次へ進む。
ゲート名とフィクスチャ名は design.md §7。
M2–M4 で作る名前は、M5 まで export しない（`checkdocs = :exports` のもとで、各段の `make docs` を通すため）。テストでは修飾名で呼ぶ。

## Milestones

### M0 — 仕様の合意と決定記録の骨子

- [x] design.md §8 の確認事項をユーザーと合意する（2026-11-21、仕様合意）
- [x] `docs/specs/constrained-mc.md`（英語）の骨子：拘束測度、受理則、null move、パートナー規則、スイープの定義と RNG 消費、レーンの seed、トルクの規約、再現性の範囲（P6）（`4edbfe5`）
- [x] `docs/specs/README.md` の両方の表（決定記録、仕様フォルダ）に登録する（`4edbfe5`）
- [x] `references/` に Asselin 2010、Toga 2016、Ellis 2017、Gong 2019 のノートを作る（`asselin2010-cmc.md`、`toga2016-weighted-cmc.md`、`ellis2017-hybrid-cmc.md`、`gong2019-ndfeb-anisotropy.md`）

### M1 — フィクスチャと 2 スピンのエネルギー差

- 新しい保証：2 スピン同時のエネルギー差は、何体項でも全エネルギー差と一致する。棄却すると zrows がビット単位で元に戻る。
- 維持する保証：`test_energy.jl` の ΔE ≡ 全エネルギー差。
- [x] §7.1 のフィクスチャ（F2g、F3、F3a、F4、FZ）。body-1 の l = 0 項で全サイトを活性にできるかも確かめる（受け付ける。全サイト活性・E 一定・勾配 0 をテストで固定）。`_poison_pair` を `fixtures.jl` に移す
  - F2u、F2w±、Fmix は M3/M4 に、FL は M8 に回す
- [x] `_CMCScratch`、`_pair_delta_energy!`
- [x] G2。変異：素朴な和、巻き戻し漏れ

### M2 — トルクと部分トルク

- 新しい保証：`total_torque` は、大域回転したときのエネルギーの微分と一致する。等方模型では 0。Zeeman の分も入る。部分トルクの和は全トルクに等しい。
- [x] `total_torque`（public）、`torque_observable`（入口の検査つき）
- [x] G3、G17。変異：符号、軸、Zeeman の漏れ・二重計上、部分 H のサイト番号のずれ

### M3 — 拘束の型・幾何・提案（純関数）・初期化

- 新しい保証：補償で M_⊥ が保たれる。受理比が独立な式と一致する。提案は可逆。不正な状態は −Inf になる。入口の検査と初期化は §2.8 のとおり。
- [x] `MagnetizationConstraint`（内部 ctor、`==` / `isequal` / `hash`）、`_ResolvedConstraint`、重みの解決
- [x] `_cmc_pair_proposal`（NaN を安全に扱う比較）、拘束付きの初期化（`_cmc_initial_config`、RNG を受け取らない）、M をゼロから計算する `_cmc_magnetization`
- [x] 補償スピン e_j′ の |z| ≤ 1 を保証（正規化してから [-1, 1] にクランプ。ヤコビアンと M∥′ は正規化後の e_j′·n̂ を使う）
- [x] G4、G0、G1 の幾何の部分（Fmix、F2w±）。変異：ヤコビアン、|z| 比、z 符号、重みの取り違え、|w| 比、実験室座標系での補償、sign(w) の欠落
  - 変異試験はメモリ上で関数定義だけを差し替えて実施（ソースは書き換えず、前後で shasum 一致）。NaN 比較の片方だけの変異は等価変異で生き残る（決定記録に理由）。両方の変異は殺せる
  - フィクスチャ F2u、F2w±、Fmix を追加。`_initial_config` の検証部分を `_validated_init` に切り出した（挙動は同一）
  - G1 の「`_renormalize!` の直後」と警告のテストは、ChainState とスイープが入る M4 で行う

### M4 — CMC スイープ

- 新しい保証：
  - 定常分布が拘束測度に一致する（G5、G6、G7）。
  - M は不変（G1）。
  - 非活性サイトと重み 0 の規約（G1b）。
  - スイープ中のアロケーションは 0（G16）。
- 維持する保証：既存のスイープには触れない。
- [x] `_attempt_cmc_pair!`、`cmc_sweep!`（直列、null move を数える、ハイブリッド、毎スイープの M 再計算、警告）
  - `cmc_sweep!` は `(nacc, M)` を返す（スイープ末の増分 M。G1/G1b でゼロからの和と比べる）
  - ずれの警告はスイープ開始時（`_cmc_check_drift`、絶対閾値）。ステップ下限と弱い拘束は `_cmc_check_step_floor` / `_cmc_check_weak_constraint` として切り出し、M5 のドライバが熱化の終わりに呼ぶ。`updates.jl` の clamp 下限に `_STEP_MIN` の名前を付けた（値は同じ）
  - (b) の決着：`_cmc_pair_proposal` が M′ をベクトルで 1 回だけ作り、4 つ目の戻り値として返す。null 判定・ヤコビアン・保存値はビット単位で同じ値
- [x] ~~`_CMCScratch` に 𝒞 の索引配列を足し~~、`length(scratch.lc) == H.nlm` を assert
  - 仕様からの変更：索引配列は `_ResolvedConstraint.cset`（M3 で導入済み）をそのまま使い、移動元の位置はサイト番号順の走査で数える（cset は昇順）。scratch に複製や逆写像は持たせない。サイズの assert は mover がまだ無いので `cmc_sweep!` の入口に置いた（O(1)）
- M3 レビューからの申し送り：
  - (a) スイープ開始時のずれの警告は**絶対**の閾値 `|M_⊥| > 1e-10·Σ_{s∈𝒞}|w_s|` で判定する。閾値近くの `init` は回転後に |M_⊥|/M_∥ が 2e-8 程度まで残るので、相対の 1e-10 では誤って警告が出る（design.md §2.5 に反映済み。`:constraint_residual` は相対のまま）
  - (b) 受理判定では M_new をベクトルとして 1 回だけ計算し、M_∥′ > 0 の判定と保存する M に同じ値を使う（不変量をビット単位で一致させる）。`_cmc_pair_proposal` は M_∥′ をスカラーの増分で出しているので、`_attempt_cmc_pair!` 側で揃える方法を決める
  - (c) 引数名：`SweepScratch` は `sc`、`_CMCScratch` は `scratch`（STYLE_GUIDE §1.5）
- [x] 統計ゲートは `cmc_sweep!` と binning を直接回す内部ハーネスで実行する（G5、G6、G7）。G6 は M5 で `run_cmc` 経由でも 1 本足す
  - `test/unit/test_cmc_sweep.jl`：G1（Fmix の dims (2,1,1) を含む）、G1b、G16、D3（1 試行とスイープ全体の RNG 消費をストリームのコピーで再生、`st.rng` は消費しない）、G0 の警告、G5、G6、G7、G5h（重み 0 のハイブリッドサイトの Langevin 則）。91 テスト、約 9 s
  - M4 レビューで追加：G5h（ハイブリッドサイトを 0.5β で試す変異が初版の 85 テストをすべて通った）、pos の数え方（`pos = s` の変異が通った）、σ の上限を 8 seed の最大 σ × 1.7–2 に設定、`_ResolvedConstraint.wnorm`、`_gpu_adapt_step!` の clamp にも `_STEP_MIN`、Logging をテスト依存に追加
  - G7 の ⟨P₂⟩ の求積は (r, R) 変数（u → R の置換で R = 0 の跳びが消える）で r = 1 で分割。(r, u) のままでは代数的にしか収束しない（n = 48→96 で 3.7e-6）
- [x] 統計ゲートの変異をメモリ上で当てて落ちることを確かめ、シフト量と σ を記録する（null move を数えないやり直しを含む）。表は `constrained-mc.md`
- [x] `bench_sweeps.jl` に CMC を加える（`.claude/bench_log.md` #5。Metropolis の約 2 倍/試行、allocs 0）

### M5 — ドライバ・観測量・API ドキュメント

- 新しい保証：`run_cmc` が、温度ラダー、焼きなまし、拘束付きの初期化と再初期化、弱い拘束の警告を行う。
- 維持する保証：既存ドライバの結果は変わらない（G15）。
- [x] リファクタ前のコミットで G15 の pin を取る（変化検出器と明記）。`bench_run.jl` と `bench_sweeps.jl` の前を計測する
  - `test/unit/test_regression_pins.jl`：手書き項の F3a（dims (2,2,1)、12 サイト）で `run_mc`（OR なし・2 温度・carryover あり）、`run_mc`（OR 2・3 温度・`carryover = false`）、`run_pt`（3 段・OR 1）。固定 seed、各点の全統計量の平均と誤差・受理率・最終ステップ・最終配置を rtol 1e-10 で比べる。約 0.03 s
  - 取得：`6cc6771` の `git archive`（同じ SCEFitting `4d2f86e`）、Julia 1.12.7、macOS 15.7 arm64、`-t 4`
  - 初版は `mc_or` が 2 温度で、再起動が 1 回しかないため「`st.rng` のコピーから引く」変異が等価変異として生き残った。3 温度に延ばして取り直した
- [x] `run.jl` に mover の新メソッドを足す（5 引数の既存メソッドは残す）。作業中に、同じ OS・同じ Julia でビット一致を一度確かめ、ここに記録する
  - **ビット一致の確認（2026-11-21、同じ Mac・Julia 1.12.7・`-t 4`）**：リファクタ後のコードが pin の全 115 項目（各点の kT・受理率 2 種・最終ステップ・全統計量の平均と誤差のベクトル、3 本の最終配置）を `isequal` で再現した。差分 0
  - `_SweepMover`（既存動作）と `_CMCMover`（`cmc.jl`）。mover が受け持つのは 1 スイープ（`_compound_sweep!(mover, …)`）、`carryover = false` の再初期化（`_mover_reinit!`）、結果型（`_mover_result`）と、熱化の終わりのフック（`_mover_after_thermalization`、最後の再正規化の直後、読み取り専用）。mover を取らない `_mc_loop!` は `_SweepMover()` を渡して委ねる（使われない `_run_temperature!` の mover なしメソッドはレビューで削除）
  - `_CMCMover` の内部 ctor で `length(scratch.lc) == H.nlm` と `length(rc.site_w) == H.n_sites` を assert
  - checkpoint：`run_cmc` は `checkpoint ≠ nothing` を「not supported yet」の ArgumentError で拒否する（M6 で入れる）
- [x] `run_cmc`（単一）、`CMCResult`（+ `show`）、`cmc_observables`、`cmc_evaluables`
  - `CMCResult.constraint` は実行時の n̂ をビット単位で保持する。公開 ctor の正規化は冪等でない（正規化済みベクトルの約 1/3 が 1 ulp 動く）ので、検証済みの値をそのまま受ける内部 ctor（`_Validated` タグ）を `MagnetizationConstraint` に足した
  - 観測量は構築に使った `H` に束縛する（`torque_observable` と同じ。別の `H` の run では ArgumentError）
- [x] bench の後を計測し、`.claude/bench_log.md` に記録する（#6）
- [x] `binning-observables.md` の B5、`docs/src/api.md`（export と同じマイルストーンで入れ、strict ビルドを保つ）
- M4 レビューからの申し送り：
  - [x] 弱い拘束の警告とステップ下限の警告の両方を `run_cmc` 経由でテストする（`test_cmc_driver.jl` の G0d。出る／出ない、温度ごと・run ごとに 1 回ずつ、kT と n̂ のキー）
  - [x] 警告の範囲：**温度ごとに 1 回**（親の決定）。`_cmc_check_step_floor(st, kT, n̂)` / `_cmc_check_weak_constraint(rc, config, kT)` は `maxlog` を外し、文面とログのキーに kT と n̂ を入れた。スイープ内のずれの警告は `maxlog=1` のまま
  - [x] `cmc_sweep!` → `_cmc_sweep!` に改名した（STYLE_GUIDE §1.6。src、bench、テスト、決定記録。design.md には「（M5 で変更）」の注記を付けた）
  - [x] `_cmc_sweep!` が返す M はスイープをまたいで持ち越さない（`_CMCMover` の `_compound_sweep!` が捨てる）
- テスト（`test/unit/test_cmc_driver.jl`、199 テスト（レビュー反映後）、約 6 s）：観測量の定義と `n_cells` 規格化（Fmix dims (2,1,1)、FZ の Zeeman トルク）、D4（再初期化が既定パターンに戻り RNG を消費しない）、ラダー・carryover・結果・`show`、G6d（F2u θ=0.6 を `run_cmc` 経由、既定の適応、seed 6101、T_x +0.58σ、T_y +0.54σ、atol 5σ）、G0d、入口。G6 の求積ヘルパーなどは `fixtures.jl` に移した
- 変異 11 種を決定記録の表に記録（2 種は初版のテストを生き残り、テストを強化した）
- M5 レビュー（2 本、blocker 0）の反映：
  - `_CMCMover` が初期配置をコピーする（`run_cmc` 側のコピーだけを外す変異が全テストを通り、2 番目の温度の ⟨E⟩ が −0.8594 → −0.7286 に動いた）。ドライバ経由のゲート D4d（エネルギー一定、step 1e-4、各温度の最初の測定配置が既定パターンから 1 スイープ以内）を追加。別名の変異は 7.5e-3（限界 1e-3）で落ちる
  - G15 は取得環境（macOS・aarch64・Julia 1.12）でだけ比べ、他では `@test_skip`。軌道は ΔE の丸め 1 つで受理判定が反転し得るので可搬でない。SCEFitting の SHA、再取得の条件（Harmonics / LegendrePolynomials の変更、Julia のマイナー版）、`_g15_print_pin()`、観測量・evaluable の明示を追加
  - API リファレンスを `docs/src/api/{index,constrained_mc,internals}.md` に分割（110 KiB の警告を解消）
  - 観測量が別の拘束で作られた場合の不一致を docstring・B5 に明記しテストで可視化、Zeeman トルクが重みでなくモーメントから作られるテスト、フックの位置のテスト（`sweeps_therm = 0`）、符号付きゼロ、`_Mover` 抽象型、`_mover_after_thermalization` への改名、`or_per_metropolis` / `sweep_tasks` を理由つき ArgumentError に、`run_cmc` の docstring（f = F/n_cells、1 回 1 拘束、`measure_interval`、例）、`show` の重みの省略とトルク列のガード、SPEC.md、公開引数名 `con` → `constraint`
- M6 への申し送り：`carryover = false` の resume には拘束付きの初期配置が要る。`init` を与えた run では他の保存内容から復元できないので、初期配置（または回転済みの `init`）をチェーンと一緒に保存すること

### M6 — Checkpoint / resume

- 新しい保証：測定途中で中断して再開しても、中断なしと一致する。拘束が round-trip する。
- 維持する保証：既存 kind の resume ゲート。
- [x] `_Checkpointer.constraint`、`_write_header`、`resume` の解決順、早期 return の分岐
  - `_Checkpointer` に `constraint`（`_constraint_record(rc)`：n̂ をビット単位、解決済みの重み、−0.0 → +0.0）と `start`（拘束付きの初期配置、3 × n_sites）を足した。`_make_checkpointer` はキーワード `constraint` / `start` を受け、"cmc" ならそろって与えられ、plan が `or_per_metropolis = 0`・`sweep_tasks = 1` であることを assert する（他の kind の呼び出しは変えていない）
  - `_write_header` だけが `constraint/{direction, weights, partner_rule = "uniform"}` と `start/config` を毎回書く。読み込みは `_read_constraint`（partner_rule を最初に確かめ、方向は `_Validated` でビット単位、単位長・有限・長さを検査）と `_read_start`（非射影の扉）
  - `_constraint_record` を `cmc.jl` に切り出し、`_mover_result`（`CMCResult.constraint`）と checkpoint の両方が同じ記録を使う
  - `resume`：既定値を `nothing` にし、schema → fingerprint → kind（"cmc" なら拘束）→ 既定値の解決 → 既存の検査の順。"cmc" では plan の固定値を検査し、mover を保存された拘束と初期配置から作り直す。早期 return は "cmc" で `_mover_result`（`CMCResult`）、"mc" は従来どおり `MCResult`。"mc" / "pt" の経路は変数名以外そのまま
  - `run_cmc` の「not supported yet」の扉を外し、`run_mc` と同じ cadence で checkpointer を渡す（温度境界の無条件の書き込みは共有の `_mc_loop!` による）
  - 仕様からのずれ（小）：初期配置は `init` を与えたときだけでなく、すべての "cmc" ファイルに保存する（読み込みの経路を 1 本にするため。既定パターンでも同じ）。group 名は `start/config`
- [x] G13（位置の assert つき）。`checkpoint-schema.md`（古い版での失敗の仕方を正確に書く）
  - `test/unit/test_cmc_checkpoint.jl`（`test_cmc_driver.jl` の後に登録、1095 テスト、約 15 s）。G13 はすべて書き手を測定途中で中断し（`_poison_pair`、既定引数の場合は毒入りの evaluable）、ファイルの位置を assert する。比較は結果・最終配置・結果の拘束・最終 checkpoint ファイルのチェーン（エネルギー、ステップ、カウンタ、チェーンとサイトごとの RNG の語）をビット単位で
    - Fmix dims (2,1,1)、3 温度、`measure_interval = 3`、`renorm_interval = 70`：carryover = true（温度 2 の測定 160 スイープ目）、carryover = false ＋ ユーザーの `init`（同じ位置。再開後に温度 3 の再初期化をまたぐ。保存された初期配置が回転した `init` で既定パターンでないことも assert）、境界だけの cadence ＋ carryover = false（温度 2/3 の境界）
    - FZ（磁場あり）、重み (1, −2)、`measure_interval = 2`、`renorm_interval = 30`（温度 2 の測定 130 スイープ目）
    - 既定引数の `resume(path, H)`（温度 1 の測定 290 スイープ目）と完了済みファイル
  - RT：`==`・`isequal`・`hash`、−0.0 の moment が畳まれて保存される、12 の乱数方向で n̂ がビット単位（少なくとも 1 つは再正規化で動く）
  - 扉：fingerprint、GPU 版の `resume`、partner_rule、明示した標準の観測量・evaluables、壊れた方向・重み・plan・初期配置
  - 既存 kind："mc" ファイルを既定引数で再開すると標準の観測量・evaluables に解決される。既存の checkpoint テストは無変更で通る
  - M6 レビュー（2 本、blocker 0）の反映：
    - 熱化途中のゲート（carryover = false ＋ `init`、温度 2 の therm 70 スイープ目＝適応の窓の途中）と二重の再開（再開した run をもう一度中断し、温度 2 の測定 110 スイープ目、再開した書き手のファイルから温度 3 の再初期化をまたぐ）。最終ファイルの比較に `start/config` と `constraint/` を加えた
    - 読み込み側の整合性の扉（非射影）：保存された n̂ と重みで `start/config` と `chain/config` の M を計算し、`M·n̂ ≤ 0` または `|M⊥| > 1e-8 Σ|w|` なら壊れたファイルとして拒否（完了済みファイルの早期 return でも）。方向を別の単位ベクトルに、初期配置やチェーンを全サイト +x に書き換えたファイルのテスト
    - 既定引数のゲートを、保存方向が正規化で冪等でない拘束でも回す（初版の修正はヘルパーが方向から拘束を作り直して再正規化し、冪等な方向になって変異が生き残った。run 自身の方向が動くことを assert）
    - `"gpu_pt"` と未知の kind を、kind を読んだ直後（既定値の解決の前）に名前で拒否。`_read_config(f, key, H)` に 3 か所の配置の読み込みを集約（メッセージにキー名）。GPU 版 `resume` も既定値 `nothing`。CPU 版 `resume` に戻り値の型注釈。`_read_constraint` の単位長の検査を `_Validated` と同じ式・同じ型で
    - `checkpoint-schema.md`：表題、重複していた `## C3` を付け直し（C3 指紋、C4 ビット一致、C5 resume、C6 cmc。参照も修正）、古い版の振る舞いを schema v3 の版（`855c17d`〜`ea4f40a`）とそれ以前に分けた。`CLAUDE.md` の coupled site に kind "cmc" の 5 行と schema v3
    - 変異 6 種を追加（5 種を殺し、`frozen` を復元しない変異は等価変異として記録）
  - 変異 11 種をすべて殺した（決定記録の表。ソースの文字列を 1 か所置き換えて `include_string` で再評価、1 変異 1 プロセス、`src/` の shasum は前後で一致。無変更の再評価は全テストを通る）
  - `checkpoint-schema.md` に kind "cmc"（C6）、C4 の kind ごとの記述（再初期化は決定論的で RNG を消費しない）、C5 の解決順、古い版での失敗の仕方（既定引数では観測量名の不一致エラー、"unknown kind" に届くのは名前の合う観測量と evaluables を渡したときだけ）。決定記録 D6、D4 の注記、ガイド `checkpointing.md`、SPEC.md（schema v2 → v3 の表記の誤りも直した）

### M7 — 拘束方向レーン

- 新しい保証：レーンの結果は `ntasks` とスケジューリングによらずビット一致し、入力順に返る。レーン k はレーン seed k の単一チェーンとビット一致する。各レーンは自分の拘束で測る。レーンごとの checkpoint は単独で resume できる。
- 維持する保証：単一チェーンの `run_cmc` の挙動（`or_per_metropolis` / `sweep_tasks` の拒否は共通ヘルパー `_cmc_refuse_unsupported` に切り出しただけで、メッセージも同じ）。
- [x] `run_cmc(H, cons; ntasks)`、レーン seed、レーンごとのファイル、`observables` を関数で受け取る
  - M5 レビューより：単一チェーンの API でも、観測量を拘束の関数として受け取る形（または `extra_observables` キーワードで既定の CMC 観測量に足す形）を検討する。今は `observables = cmc_observables(H, con2)` を `con1` の run に渡すと `:M_par` / `:constraint_residual` が黙って con2 に沿って測られる（docstring に明記済み）
    - **決着（M7）**：単一チェーンは合意どおり `observables::Vector{Observable}` のまま。レーン版だけ関数 `constraint -> Vector{Observable}` で受け取り、Vector を渡すと理由つきの ArgumentError。**`extra_observables` キーワードは v1 では足さない**（`c -> [cmc_observables(H, c); extra]` で足りる。決定記録 D4、D7）
  - 観測量の関数は、レーン順に呼び出し側のタスクで 1 回ずつ、どのレーンも走る前に呼ぶ（戻り値の型、`_check_observables`、`_check_evaluables` もそこで検査し、例外は包まれずに出る）
  - `evaluables` は全レーン共有の Vector（ビン統計だけを読み、拘束に依存しない。`Evaluable` は不変）
  - 重複した拘束は許す（レーン seed が違う独立な複製）
  - スケジューリング：レーンを連続したブロックに分け（`Iterators.partition(1:K, cld(K, nt))`）、ブロックごとに `Threads.@spawn`。`ntasks = 1` は呼び出し側で直列。結果はスロット k にレーン k だけが書く。連続ブロックにしたのは、seed をタスク内で遅延に引く変異がどのスケジュールでも必ず検出されるようにするため（共有カウンタだと開始順がたまたまレーン順になりうる）
  - 仕様からの追加：1 レーンの失敗で他のレーンは止めない。全レーンを回してから、失敗をレーン順の `CompositeException`（中身は `CapturedException`）で投げる（`ntasks` によらず同じ形。完了したレーンのファイルは完全）。`InterruptException` はすぐ投げ直す。初版は `@sync` のままで、`ntasks = 4` では失敗したレーンと同じブロックの後続レーンが走らず、どのレーンが走るかが `ntasks` に依存した（初版のテストで発見）
  - 入口の扉（空の拘束ベクトル、`ntasks < 1`、負の seed、Vector の observables、関数の戻り値の型、`or_per_metropolis` / `sweep_tasks`、観測量と evaluables の検査、H に解決できない拘束）は、どのレーンも走る前に投げる。それ以外のキーワードの誤り（`temperature` と `kT` の両方など）はレーンから `CompositeException` で出る
  - checkpoint：`<stem>.lane<k>.jld2`（`.jld2` で終われば除いたもの、そうでなければパスそのもの）。各ファイルは普通の "cmc" ファイル（レーン seed を `plan/seed` に持つ）。集合全体の resume は無い（O6、決定記録 D7）。`checkpoint-schema.md` C6 と guide `checkpointing.md` に 1 段落
- [x] G14（`test/unit/test_cmc_lanes.jl`、`test_cmc_checkpoint.jl` の後に登録。3477 テスト、約 6 s）
  - Fmix dims (2,1,1)、6 レーン（Fmix の重みで 4 方向、レーン 2 の重複、`:uniform` のレーン）、2 温度
  - `ntasks` 3、4、20 ≡ `ntasks = 1`（結果の全フィールド、最終配置、拘束、seed を `==`）
  - レーン k ≡ レーン seed k の単一チェーン。seed はテスト側で規則から導出（パッケージのヘルパーは使わない）
  - 各レーンが自分の拘束で測る：全温度で `:constraint_residual` < 1e-13、`:M_par` > 0。最終配置の重み付き M（テストで和を取る、レーンの重みで）がそのレーンの n̂ に平行で、他のレーンの n̂ には平行でない。重複レーンは元と違う
  - checkpoint：ファイル名（`.jld2` あり・なし、余分な名前なし）、kind "cmc"、方向と重み、完了したレーンの `resume(file_k, H)` ≡ レーンの結果
  - 中断：レーン 3 だけに `_poison_pair`、ファイルは温度 2 の測定 40 スイープ目（assert）。`ntasks` 1 と 4 で resume ≡ 中断なし。他のレーン（`ntasks = 4` でレーン 3 と同じタスクのレーン 4 も）は完走。失敗は 1 要素の `CompositeException`
  - 扉：上の一覧。どの扉でも checkpoint ファイルができないこと
  - 変異 8 種をすべて殺した（決定記録の表。遅延 seed、観測量 Vector の共有、完了順の収集、ファイル番号のずれ、stem 規則の無視、seed の 1 つずれ、失敗でブロックが止まる、Vector の扉の除去）。ソース文字列を置き換えて `include_string` で再評価、1 変異 1 プロセス（`-t 4`）、無変更の再評価は全テストを通る、`src/` の shasum は前後で一致
- [x] M7 レビュー（2 本、blocker 0）の反映
  - 中断：`Threads.Atomic{Bool}` の停止フラグを各ブロックがレーン開始前に見る。レーン内の `InterruptException` でも、待機中の呼び出し側への Ctrl-C でも立て、呼び出し側は全タスクの終了まで待ってから裸の `InterruptException` を投げる（`_cmc_wait_lanes`。タスクが呼び出しより長生きして checkpoint を書き続けることがない）。走っているレーンは途中で止めない（docstring と D4）。初版は `ntasks ≥ 2` で中断が自ブロックしか止めず、呼び出し側の Ctrl-C で spawn したタスクが残った
  - 入口の検査をレーンより前に全部：単一チェーンのキーワードを `run_pt` と同様に明示（綴り違いは呼び出し時の MethodError）、`resolve_kt` を 1 回だけ呼んで全レーンに `kT = kts`（ビット同一、一度きりの反復子も 1 回だけ消費）、`UpdatePlan` を 1 回作って計画の扉、全レーンの `_resolve_constraint` と `_cmc_initial_config(H, rc, init)` を観測量関数と無関係に実行、checkpoint のパスと `checkpoint_interval`
  - 失敗の報告：レーンごとに `@error "run_cmc: lane k (n̂ = …) failed; the other lanes go on"`（例外とバックトレース付き）。レーン順の `CompositeException`（`CapturedException`）はそのまま。公開の例外型は足さない。docstring に回復手順（長いスキャンは `checkpoint`、完了したレーンのファイルは `resume` で結果を返す）と、古いレーンファイルが残りうること（最初の書き込み前に失敗したレーン、少ないレーンでの再実行）。guide `checkpointing.md` と D4 にも
  - master は `Xoshiro(UInt64(seed))`（`run_pt` と同じ）。`ntasks::Union{Nothing,Integer} = nothing` → `min(K, nthreads())`、文言は「タスク数 ≤ ntasks、`cld(K, ntasks)` レーンずつの連続ブロック」
  - 扉：メッセージの接頭辞 "run_cmc:"、呼び出せない値（タプル）に専用の ArgumentError、docstring に `c -> obs` が扉をすり抜けること、自前の evaluables が可変状態を持つと `ntasks ≥ 2` で競合すること
  - `_lane_checkpoint_stem`：`.jld2` は大文字小文字を問わず 1 回だけ外す。ディレクトリを指すパス（空、`/` 終わり、既存ディレクトリ）と空の stem（`.jld2`）を拒否
  - 単一チェーンの docstring にレーン版のシグネチャを明記。`plan = UpdatePlan(` の空白を復元
  - テスト：ntasks 非依存ゲートを (2, 4, 20)（3+3、2+2+2、1 レーンずつ）に。AGG（レーン 1 が最後の測定で、レーン 6 が最初の測定で失敗、ntasks 1/4/20、レーン順、全要素が `CapturedException`、`@error` 2 件。`ntasks ≥ 2` ではレーン 6 の失敗までレーン 1 を待たせて完了順を逆にする。初版の AGG は 6 タスク・4 スレッドでレーン 6 がレーン 1 の後に始まり、完了順の変異が生き残った）。INT（レーン内の中断 ntasks 1/4、呼び出し側への中断 ntasks 4、裸の `InterruptException`、走っていたレーンだけがファイルを持ち、`run_cmc` が投げた時点で完全）。扉の追加（temperature/kT、計画、綴り違い、タプル、観測量関数と無関係な解決、1 レーンだけの `init` の扉、checkpoint のパス、一度きりの反復子）。3526 テスト、約 9 s
  - 変異：初版の 8 種に、レビューで生き残った 4 種（失敗の完了順、逆順、`CapturedException` なし、中断の再送出の削除）と新しい扉・中断の 10 種を加えた計 21 種をすべて殺した（決定記録の表）。`src/` の shasum は前後で一致
  - テストの決定論化（コーディネーター指示）：INT と AGG の順序付けを `sleep` から明示的なハンドシェイクに置き換えた。レーンは測定 1 回目に原子カウンタで「開始」を報告し、走っているレーンはパッケージ自身のログ（失敗の `@error` は記録の後、中断の `@warn` は停止フラグを立てた後に出る）をテストのロガーが見て立てるフラグまで待つ。呼び出し側への中断は、全レーンの開始を確認し、さらに呼び出し側が待ち行列にいる（内部フィールド `Task.queue`。全レーンが待機中なので抜けられない）ことを確認してから別タスクが `schedule` する。待ちには上限（30 s）を付け、合図を出さない変異はハングでなく失敗になる。そのためにパッケージ側に中断時の `@warn`（1 呼び出し 1 回、`_cmc_interrupt!`）を足した（「新しいレーンは始めない、走っているレーンは終わるまで待つ」と利用者に伝えるもので、D4 と docstring に記載）。`-t 4` で 5 回連続、`-t 2` で 1 回、全部通過（3527 テスト）

### M8 — docs・手動検証・規約ファイル・パネル

- [x] `docs/src/guide/parallelism.md` に CMC レーンの段落（M7 レビューより）
  - 決定論（seed はレーン順に先に引く、`ntasks`・スレッド数によらずビット一致）、連続ブロック、重複拘束＝複製、失敗の集約、中断の意味、レーンごとのファイル。「Choosing the layer」の表に 1 行
- [x] guide `docs/src/guide/constrained_mc.md`（K の取り出し、部分トルクとその解釈上の注意、重みの選び方 = FePt・強磁性・Nd₂Fe₁₄B・反強磁性、有限 N の注意）。theory、`docs/make.jl`、SPEC.md、README、CHANGELOG
  - guide の `@example`：手書きの sc 強磁性体 4×4×4（等方 J = −1、z 結合の e_z e_z′ 二イオン項、l = 2 の 1 イオン項）、θ 7 点 × 複製 2 のレーン、kT = 0.7、500 + 4000 スイープ。トルクの符号と規格化、複製のプール（χ²/dof）、一軸の重み付き最小二乗（K1、K2）、θ = π/4 の 1 点推定、Simpson 積分による整合性、部分トルク（和の規則、1 イオン ≈ m³・二イオン ≈ m² の比較）。走査は 2.3 s（コンパイル込み、2 回目 0.6 s）、ページ全体の実行は約 3 s
  - 本文：κ_lm の一般のフィット（research/A-theory を参照）、空の部分集合の ctor エラー、`red.terms`、O(K) の注意、重みの選び方（`:auto`、FePt の Pt ハイブリッド、フェリ磁性体と Nd₂Fe₁₄B の符号付き重み vs Fe のみ、反強磁性体と弱い拘束、Néel ベクトルが表せない場合）、`find_ground_state` の `init`、`measure_interval`、複製による誤差、有限 N、レーンごとの checkpoint、未対応の機能、文献 4 本
  - theory `docs/src/theory/constrained_mc.md`：M∥² 因子と F の意味、ペア移動と受理則の導出（ファイバー密度）、null move の数え方（π(1 − r)）、パートナー規則、ゲージ写像、スイープとハイブリッド則、直列である理由、トルク恒等式と成分、部分トルク、線形応答。決定記録へのリンク
  - `docs/make.jl` と `docs/src/index.md` に 2 ページを登録、`docs/src/api/constrained_mc.md` から guide と theory へリンク。README に機能の 1 行。CHANGELOG の Added（CMC 一式）と Changed（`resume` の既定値と戻り値型、GPU 版の `nothing`、API リファレンスの 3 分割）。SPEC.md は M5–M7 で更新済み（変更なし）
  - `make docs`（strict）：新しい警告なし（既存の example サイズ閾値と deploy の通知のみ）、全体 71 s。`make test-unit`：24836 件すべて通過（67 s）
- [x] `bench/validate_cmc_*.jl`：G8、G9、G11 を実行して記録する
  - 2026-11-21、HEAD `c44f764`（スクリプトは未コミットで実行、`src/` は無変更）、SCEFitting `4d2f86e`、Julia 1.12.7、M4 Mac、`-t 4`。結果の表は決定記録の「Manual validation」。共通ヘルパーは `bench/validate_cmc_common.jl`、使い方は `bench/README.md`
  - G8（`validate_cmc_manybody.jl`）：回転 iid 参照（2×10⁷ サンプル、トルクは `total_energy` の中心差分）vs `run_cmc` 4 レーン × 2×10⁶ スイープ。F3（奇）、F3 の重み (1, −0.6, 1.4)、F4（偶）、F4o（ls = [1,1,1,2]、奇。スクリプト内で定義）を ±n̂ で。28 行すべて 4σ 以内（最悪 1.86σ）、分解能 ≤ 0.66 %。奇の模型の ⟨T⟩(n̂) − ⟨T⟩(−n̂) は 1085σ / 377σ（漏れが見える）、偶の F4 は 1.0σ。約 4 分
    - 分解能の基準は小さな試走の後、本番の前に |⟨ê_φ·T⟩| から |⟨T⊥⟩| に変えた（F4o の +n̂ で ê_φ 成分がたまたま小さい）。スクリプトの冒頭に記載
  - G9（`validate_cmc_linear_response.jl`）：FL（sc 4×4×4、K = 1e-3）、θ = π/4、kT/|J| = 1.4、1.2、0.7。K_eff/K と `run_mc` の ⟨P₂⟩_iso が +0.05 %、−0.13 %、−0.03 %（+0.24σ、−1.09σ、−0.89σ）、σ_comb/⟨P₂⟩ ≤ 0.21 %。約 2.7 分
  - G11（`validate_cmc_histogram.jl`）：sc 3×3×3 + K（βK = 0.28、パイロットで βΔF ≈ 3 に設定）、17 レーンの Simpson vs `run_mc` のヒストグラム（jackknife の Evaluable）。8 点すべて 4σ 以内（最悪は端点の −2.52σ、βΔF = 2.7511(41) vs 2.7402(15)）。ヒストグラムが 7/8 点でわずかに上（ビン幅の 2 次の偏り ≈ +0.003 kT と同程度）。約 2.4 分
- [x] G12 の本番スモーク（`validate_cmc_smoke.jl`）
  - l02 は等方模型（配置ごとにトルク 0）なので、それだけでは盲目。`fit_mfa.jl` の手順で l02 を再フィット（7 月のログと R²(E)、rmse(T) が一致。`pair_cutoff` は `cutoff` に読み替え）し、同じ結晶から SCEFitting で作った P4₂/mnm 対称の合成異方性（Nd の body-1 l = 2 で +30 meV/cell・容易軸 c、Fe–Fe の l = 1 ⊗ 1 対 < 2.6 Å で 10 meV/cell）を足した
  - 前提の検査：タイルした H が C4z、C2x、C2y（m_y のスピンへの作用）で 3e-14 まで不変（スピンを回さない対照は 0.7–3 % 変わる）
  - dims (2,2,2)、a–c 面の θ = 0..π（9 点）× 重み 2 種（`:auto`、Fe のみ = Nd ハイブリッド）、600 K → 300 K、3000 + 30000 スイープ、seed 1204 と 2204。両 seed で全検査に合格（⟨n̂·T⟩・⟨ê_θ·T⟩ = 0、τ(0) = τ(π/2) = τ(π) = 0、τ(θ) + τ(π − θ) = 0、閉路積分 = 0、部分トルクの和 = 全トルク ≤ 1e-14、残差 ≤ 7e-14、受理率 0.49–0.53）。1 レーン・1 スイープ約 1.2–1.3 ms（512 活性サイト、5 スイープごとにトルク 3 本）
  - τ(π/4) は 300 K で −11.9、600 K で −4.1 meV/cell（T = 0 の一様状態は −40）。2 種の重みは誤差内で一致
  - 統計の注意：seed 間の差から見ると 6000 測定の binning 誤差は約 20 % 小さい（非零 τ の 24 レーンで χ²/dof = 1.57）。seed 1204 の閉路積分 4 本がそろって約 2σ だったので seed 2204 を追加し、符号がそろわないことを確かめた
- [x] `CLAUDE.md`（Coupled code sites と Standing contracts）、`.claude/agents/*.md` の更新
  - `CLAUDE.md`：Project goal に CMC、Coupled code sites に「Constrained MC ↔ the single-site machinery」（D3 の RNG 順、`_pair_delta_energy!`、mover 層、kind "cmc"、ゲート一覧と G15）、Tests に手動検証スクリプト（CI 外）、Standing contracts に `constrained-mc.md`、Performance の hot path に CMC スイープ
  - agents：test-runner（`test_cmc_*.jl` 7 本と `test_regression_pins.jl` の行、schema v2 → v3、失敗の読み方）、numerical-reviewer（決定記録の一覧、CMC の観点、決定論）、code-reviewer、maintainability-reviewer（mover 層）、performance-reviewer（CMC スイープの hot path）、api-reviewer（新しい公開名、API ページの分割）、spec-reviewer（決定記録の一覧、cmc.jl に触れる仕様の確認事項）
- [x] Tier 2 の 4 軸パネル
  - ステップ A（挙動中立の指摘、2026-11-21 適用）。結果はビット単位で不変（同じ演算順・同じ RNG 消費）。`make test-unit` は前後とも 24836 件通過、G15 の pin とビット一致ゲートは無変更で通過。加えて、作業前の HEAD `773b7df` と作業後で `run_cmc`（ラダー、`carryover = false` + `init`、Fmix の符号付き・重み 0・非活性、磁場つき + 部分トルク、レーン、checkpoint/resume）と `run_mc`（OR・磁場つき `:M` / `:M_B`）の全統計量・最終配置のビット列を比較して一致
    - `src/cmc.jl`（1313 行）を `src/cmc/constraint.jl`・`pair_move.jl`・`driver.jl`・`lanes.jl` に分割。`_check_cmc_config` を checkpoint.jl から constraint.jl へ。`torque_observable`・`_zeeman_torque`・`_term_key` は CMC の型を使わないので `observables.jl` の `total_torque` の隣へ（include 順で問題なし）
    - 重複の整理：`_cmc_setup`（単一鎖とレーンの事前検査で共有）、`_check_checkpoint_interval`（checkpoint.jl と共有）、レーンの汎用スケジューラ `_run_lane_blocks(run_lane!, lane_name, nlanes, nt)`（停止フラグと中断フラグはスケジューラ内部に持たせたので、引数は指摘の `stop` の代わりに失敗ログ用の `lane_name`）、`_n_cells(H) = prod(dims)`（`n_sites ÷ n_cell_atoms` と整数として等しい）、`_moment_total`（`:M`・`:M_B`・Zeeman トルクで共有）、`_bound_to`、定数 `_CMC_INIT_MIN_M`・`_CMC_ANTIPARALLEL_TOL`・`_CMC_RESTORE_TOL`、`_is_unit_direction`、`_ResolvedConstraint` のフィールド名（`direction`・`cell_weights`・`site_weights`・`constrained_sites`・`weight_norm`）、レーンの補助関数を `_cmc_lane_*` に統一、mover のフックを `_mover_sweep!` に改名（PT の 5 引数 `_compound_sweep!` はそのまま）、mover なしの `_mc_loop!` を削除、`st.step = plan.step0` を両 mover から共有の `_restart_chain!` へ（順序は同じ）、`resume` の "mc" / "cmc" を 1 経路に
    - 注意点：`_moment_total` を外に出しただけだと `:M_B` が約 30 % の配置で 1 ulp 動いた（呼び出し側の `dot` の積和の縮約が関数境界で変わる）。`@inline` にして元のビット列に戻し、20 万配置で旧実装と一致を確認。この差は単体テストでは検出されず、上のビット比較で見つかった
    - テスト：マイルストーン番号を除去、testset 名を「CMC <topic>」とゲート記号つきに（`G6d:`・`G0d:`・`D4d:`・`OBS:`・警告は `W0:`）、共通の補助関数とフィクスチャを `test/unit/fixtures_cmc.jl`（接頭辞 `_cmcref_`）へ、torque/energy の接頭辞なし補助関数に `_tq_` / `_en_`、G15 に固定コピーの `_g15_terms()`（`_f3a_terms` と全フィールド一致を確認）、`SCEMC_TEST_VERBOSE` を test-runner に記載
    - 決定記録を常設の契約に（Status、マイルストーン表記の除去、「Gates」索引、research への参照を 1 文の oracle に、D5 の「per training cell」を修正）。bench の検証スクリプト（research 参照の除去、92 桁、未使用の `pool_proj` を削除）、guide のリンク、SPEC.md・CLAUDE.md・agents の更新
  - ステップ B（ユーザー承認済みのパネル指摘、2026-11-21 適用）。既存の観測量・統計量はビット単位で不変：HEAD `93257b7` を `git archive` で展開した版と作業ツリーで、7 本の run（単一鎖、磁場 + `:magmoms`、符号付き重み + `carryover = false`、レーン 4 本、θ = 0 と π を含む）の既存観測量（`:energy`・`:energy2`・`:torque`・`:M_par`・`:constraint_residual`・`:torque_zeeman`・`:specific_heat`）の平均・誤差・τ・件数、受理率・最終ステップ・最大ずれ・最終配置のビット列ダイジェストが一致。`make test-unit` は 24836 → 25989 件、JET・Aqua・`make docs`（strict）通過
    - `:torque_frame`：`(ê_φ·T, ê_θ·T, n̂·T)/n_cells`。`:torque` と同じ 1 回の勾配評価を共有（`_TorqueMemo`：配置がサイトごとに `===` で一致したときだけ再利用、ロック付きなのでレーン間で共有されても正しい）。枠は `cmc_observables` で 1 回だけ作る（`ê_φ = normalize(ẑ × n̂)`、`ê_θ = ê_φ × n̂`、手計算で標準の球面単位ベクトルと一致、`(ê_θ, ê_φ, n̂)` が右手系）。極（`|ẑ × n̂| < 1e-12`）では `ê_φ = ŷ`（n̂ に対して直交化）、北極で `ê_θ = x̂`、南極で `−x̂`（どちらも φ = 0 の極限）。射影の誤差は `:torque` の成分ごとの誤差からは作れない（共分散を持たない）ので、射影そのものを binning する
    - 開始時の検査 `_check_cmc_observables`：`:M_par`・`:constraint_residual`・`:torque_frame` の関数を `_ConstraintBound`（作った `H` と解決済み拘束を保持）にし、`run_cmc`（拘束の解決直後、checkpointer より前）、レーンの事前検査（`c -> 固定ベクトル` も全レーン開始前に拒否）、`resume`（ファイル自身の整合性検査の後）で比較。方向は弦 `1e-6`、重みはサイトごとに `1e-6 max|w|`（非活性原子の重みは比べない）。設計の変更点：指示の「開始配置で残差を評価する」案ではなく、保持した拘束を直接比べる方式にした（既定の開始配置は正の重みなら常に `M ∥ n̂` なので重みの不一致が見えない、`−n̂` 用の観測量は残差 ≤ 0、復元されたチェーンは `|M⊥|` が D6 の上限まで許される、の 3 点で残差方式は穴がある）。閾値の根拠は決定記録 D5
    - API と文言：弱い拘束の警告に「重みは単位胞原子ごと、磁気胞より小さく `reduce_cell` しない」、`:auto` の docstring（何に解決されるか、弱い拘束で警告）、`CMCResult` の表の列を `torque_x/y/z` にして末尾に `:torque_frame` への注記、`MagnetizationConstraint` の位置引数の内部コンストラクタをタグ `_Checked` の裏へ（`_Validated` はそのまま、checkpoint 読み込みは無変更）、`torque_observable` の拒否メッセージに引数順 `(name, Hpart, H)`
    - 数値の文言：guide の `K₂` は異方性の 2 次の本物の効果（`K₂/λ²` 一定）、Nd₂Fe₁₄B の「20 % 過小」を χ²/dof 1.57（p ≈ 0.04）の ~2σ の示唆に改め、本番の誤差は複製レーンから、`m³`（Callen–Callen）は目安で正確な O(K) の基準は `⟨P₂⟩_iso`、D5・B5・theory の per-cell 表記（`n_sites ÷ n_cell_atoms`、H 自身の胞）
    - テスト：G18（枠の閉形式・極・右手系、各成分の剛体回転差分、一軸の閉形式、勾配の共有と memo の無効化、`run_cmc` 経由で標本ごとに `:torque` の射影と一致）、G19（単一鎖・レーン・resume の各入口で不一致は拒否、一致・1 ulp 違い・`:auto` vs `:magmoms`・非活性原子の重み違いは通る、共有セットの並列レーンが個別セットとビット一致）。変異 6 種（外積の順序、`ẑ × n̂` の符号、極の分岐の除去、古い memo、共有なし、検査の除去）をすべて殺した（決定記録の表）。極の分岐の除去は初版のテストを通った（Irrational の `π` は `sin(π) == 0.0` で極そのもの）ので Float64 の π を加えた
  - ステップ C（パネル指摘 O5：トルク測定の費用。ユーザー判断「まず測る、`measure_interval` の既定は 1 のまま、docstring の助言を強める」、2026-11-21 適用）。記録は `.claude/bench_log.md` #8、決定記録 D5「The chain-state torque path」「Measurement cost」
    - 測定（Nd₂Fe₁₄B 本番模型 l02・l044_c4.4、4×4×4 = 4352 サイト）：勾配 1 回 = 拘束スイープの 0.45–0.52 回分。`measure_interval` = 1 / 5 / 10 で測定の割合は既定セット約 33 % / 9–10 % / 5 %、部分トルク 2 本（模型を覆う分割）付きで約 53 % / 17 % / 9 %。1 測定あたり 290–450 KiB（部分トルク付き ×3）の割り当て、GC 時間は ≤ 1 %。共有セットを 2 レーンで使うと memo の取りこぼしで l02 で +20 %
    - 判断：内部経路を実装。費用の大半は勾配そのもの（キャッシュでは消えない、助言が主な手段）だが、行の再構築と割り当てが軽い対模型（l02）では測定の 21–25 %、1 本 1.7 GiB のゴミが出る、memo のロックと共有セットの取りこぼしを消せる、の 3 点
    - 実装：`_TorqueObservable`（`:torque`・`:torque_frame`・`torque_observable`）。公開呼び出し `(config, E, H)` は従来どおり `total_torque`（`run_pt`・利用者のラッパ・直接呼び出し）。`run_mc` / `run_cmc` の測定は `_measure_chain!` → `_observe` → `_chain_torque!` でチェーンの `st.zrows` と鎖ごとの `_MeasureScratch`（測定ごとの `stamp`、ハミルトニアンごとのスロット）から割り当てなしで 1 回だけ計算し、`:torque` と `:torque_frame` が共有。`_TorqueMemo` は削除（ロックなし、共有セットでも取りこぼしなし）。部分トルクは `H` の行の前半を読む（行の値は `lmax` に依らない）。直接呼び出しでは `:torque` と `:torque_frame` が別々に勾配を計算する（値は同じ）
    - 前後（同一プロセス A/B）：l02 の測定 −21 %（部分トルク付き −25 %）、l044 −3 %（−4.5 %）。割り当て 295 KiB → 736 B（戻り値の箱詰めのみ）。結果はビット単位で不変：両模型の測定ラン（k = 1/5/10、レーン 1/2/4、共有セット）と l02 2×2×2 の 12 本（ラダー、部分トルク、磁場 + `:magmoms`、符号付き重み + `carryover = false` + `init`、重み 0 + 極、レーン、共有セット、checkpoint と resume、`run_mc`、`run_mc`・`run_pt` + CMC セット）を `git archive HEAD`（+ Manifest）と比較して一致
    - テスト：G18 の memo 部分を書き換え（スロットの評価回数で共有と次の測定での再計算を確認、公開呼び出しとビット一致）、G20 を追加（`_chain_torque!` ≡ `total_torque` をビット単位で、F2g・FZ・F3a・F4・Fmix、全体と部分ハミルトニアン、乱数配置 20 と拘束スイープ 30 回 + 再規格化、評価回数、割り当て 0；`run_cmc` 経由で既定セット + 部分トルク 3 本の統計量が素のクロージャ経由とビット一致）。変異 7 種をすべて殺した（決定記録の表）
    - 文書：`run_cmc` の `measure_interval`、`cmc_observables`、`torque_observable` の docstring、guide の「Thin the torque」と部分トルクの費用、D4（レーンで可変状態を持たない）、D5、Gates（G18・G20）、CHANGELOG（Performance）、SPEC.md、CLAUDE.md（`_site_grad` の連動箇所に `_torque_from_rows`、CMC の項）
  - 着地前の修正（2026-11-21、`773b7df..1ac5601` への 2 本のレビュー、blocker 0）。実行の統計量はビット単位で不変（追加したのは開始時の検査と文書のみ）：`git archive HEAD`（`1ac5601`）+ Manifest と作業ツリーで l02 2×2×2 の 12 本のダイジェスト（bench_log #8 と同じスクリプト）が一致
    - 文書：決定記録になかった memo の記述を整理（B5、レーンの `run_cmc` docstring、レーンのテストのコメント、D5 は「不採用案」として 1 段落に）、CHANGELOG の Performance 項目を `:torque_frame` の項目に畳み、最終の挙動（チェーン状態の経路、値は公開呼び出しとビット一致）だけを書く。レーンの docstring で方向依存の観測量に `:torque_frame` を追加
    - 開始時の検査の追加 2 点：(1) `_TorqueObservable`（`:torque`・`torque_observable`）が別の `H` に束縛されていれば拒否（部分トルクは `Hpart` ではなく束縛した全体の `H` を比べる）。以前は最初の測定で初めて落ちた（熱化と checkpoint の書き込みの後）。(2) `:torque_frame` は枠の軸 `ê_φ`・`ê_θ` も各 `1e-6`（弦）で比べる。極の近くでは方向が許容内でも枠が O(1) 入れ替わる（`MagnetizationConstraint(0.0, 0.0)` の run に `(5e-7, π/2)` の観測量 → 接成分が入れ替わる）。D5 の「`≤ 1e-6 rad`」の主張を正し、「なぜ 1e-6」に重みの側（1 次：残差と `:M_par` の相対ずれが最大 ~`1e-6 Σ|w|/M∥`、弱い拘束の境界で ~2e-5）を加えた
    - テスト：G19 に torque の `H` 不一致（`run_cmc`・レーン・`resume` の途中と完了ファイル）、極での枠の不一致（両向き）、角度とベクトルで作った同じ方向（極を含む）が通ること。G20 に `_measure_chain!` がチェーン状態の経路を通ること（スロットの評価回数）、`carryover = false` の 2 温度・7 スイープごとの再規格化・測定途中の中断からの resume（2 経路を交差）での統計量のビット一致。変異 3 種（`H` 検査の除去、枠の比較の除去、測定を公開呼び出しに）をすべて殺した（決定記録の表）
    - `.claude/agents/test-runner.md` に G18・G19・G20 の行、`pt-threads-determinism.md` P6 に `run_cmc` の 1 行
- [x] M6 からの申し送り：CHANGELOG に `resume` の公開シグネチャの変更（`observables` / `evaluables` の既定値が `nothing`、戻り値に `CMCResult`、GPU 版も `nothing`）を明記する。`.claude/agents/test-runner.md` の schema v2 の記述と、`test_cmc_*.jl` の行を更新する

## Exit checklist

Run through every item once implementation lands. ~~Strike through~~ items
that do not apply.

- [x] `make test-all` passes (4 threads).
- [x] `make docs` builds (strict).
- [x] If results or a determinism contract changed: test added with an
      implementation-independent oracle; bitwise gates still hold.
- [ ] ~~If `src/gpu/` changed: device validation on a CUDA node recorded in
      `.claude/bench_log.md`.~~
- [x] If public API changed: `SPEC.md` and `docs/src/api.md` updated;
      dependent-package-facing names flagged. (The API pages are
      `docs/src/api/`; the new exports are on `api/constrained_mc.md`.)
- [x] Decision record(s) under `docs/specs/` updated.
- [x] If a hot path was touched: before / after recorded in
      `.claude/bench_log.md`.
- [x] Tier 2 review panel run and findings resolved.
- [x] If module names or Makefile targets changed: `.claude/agents/` swept.
- [x] `CHANGELOG.md` `[Unreleased]` updated.
- [x] `Status:` line here and the table in `docs/specs/README.md` updated.
- [x] Implementation commit hash appended below.

`make test-ci`（`test-all` + strict `make docs`）は着地前の修正の作業ツリーで通過：26395/26395（`1ac5601` の 26056 から +339）、strict docs 通過。

Implementation commits: `4edbfe5` (spec and decision record), `91e1448`,
`f6ef95c`, `6cc6771`, `ea4f40a`, `b1de3bc`, `c44f764`, `191fc83`, `773b7df`,
`93257b7`, `377cfdf`, `1ac5601`, and the landing-fix commit that follows
`1ac5601`.
