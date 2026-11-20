# Tasklist: constrained Monte Carlo (CMC) for finite-temperature anisotropy

Status: agreed (2026-11-21) — M4 完了、M5 へ

このファイルには、コミット単位の大まかなマイルストーンを置く。日々の追跡はセッション内の `TaskCreate` で行う。

各マイルストーンでは、実装の前に「新しく宣言する保証」と「維持する保証」を挙げる。
新しい保証には、落とすべき誤りと、それが見えるフィクスチャを添える。
実装が終わるたびに、別エージェントのレビュー（numerical-reviewer。独立な変異試験を含む）を通してから次へ進む。
ゲート名とフィクスチャ名は design.md §7。
M2–M4 で作る名前は、M5 まで export しない（`checkdocs = :exports` のもとで、各段の `make docs` を通すため）。テストでは修飾名で呼ぶ。

## Milestones

### M0 — 仕様の合意と決定記録の骨子

- [ ] design.md §8 の確認事項をユーザーと合意する
- [ ] `docs/specs/constrained-mc.md`（英語）の骨子：拘束測度、受理則、null move、パートナー規則、スイープの定義と RNG 消費、レーンの seed、トルクの規約、再現性の範囲（P6）
- [ ] `docs/specs/README.md` の両方の表（決定記録、仕様フォルダ）に登録する
- [ ] `references/` に Asselin 2010、Toga 2016、Ellis 2017、Gong 2019 のノートを作る

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
- [ ] リファクタ前のコミットで G15 の pin を取る（変化検出器と明記）。`bench_run.jl` と `bench_sweeps.jl` の前を計測する
- [ ] `run.jl` に mover の新メソッドを足す（5 引数の既存メソッドは残す）。作業中に、同じ OS・同じ Julia でビット一致を一度確かめ、ここに記録する
- [ ] `run_cmc`（単一）、`CMCResult`（+ `show`）、`cmc_observables`、`cmc_evaluables`
- [ ] bench の後を計測し、`.claude/bench_log.md` に記録する
- [ ] `binning-observables.md` の B 項、`docs/src/api.md`（export と同じマイルストーンで入れ、strict ビルドを保つ）
- M4 レビューからの申し送り：
  - [ ] 弱い拘束の警告とステップ下限の警告の両方を `run_cmc` 経由でテストする（条件を満たすと出る／満たさないと出ない）。ドライバが熱化の終わりに `_cmc_check_step_floor` / `_cmc_check_weak_constraint` を呼ぶ
  - [ ] 警告の範囲を決める：`maxlog=1` は呼び出し箇所ごと・ロガーごとなので、温度ラダーや複数回の `run_cmc` では最初の 1 回しか出ない。kT と n̂ を警告の kwargs に入れる（`_id` を分ける）か、1 run に 1 回と決めるか
  - [ ] `cmc_sweep!` を `_cmc_sweep!` に改名するか決める（STYLE_GUIDE §1.6：内部名は `_` で始める。design §8 で内部にとどめると決めている）
  - [ ] `cmc_sweep!` が返す M はスイープをまたいで持ち越さない（毎スイープの再計算が resume ≡ 中断なしの前提）

### M6 — Checkpoint / resume

- 新しい保証：測定途中で中断して再開しても、中断なしと一致する。拘束が round-trip する。
- 維持する保証：既存 kind の resume ゲート。
- [ ] `_Checkpointer.constraint`、`_write_header`、`resume` の解決順、早期 return の分岐
- [ ] G13（位置の assert つき）。`checkpoint-schema.md`（古い版での失敗の仕方を正確に書く）

### M7 — 拘束方向レーン

- [ ] `run_cmc(H, cons; ntasks)`、レーン seed、レーンごとのファイル、`observables` を関数で受け取る
- [ ] G14

### M8 — docs・手動検証・規約ファイル・パネル

- [ ] guide `docs/src/guide/constrained_mc.md`（K の取り出し、部分トルクとその解釈上の注意、重みの選び方 = FePt・強磁性・Nd₂Fe₁₄B・反強磁性、有限 N の注意）。theory、`docs/make.jl`、SPEC.md、README、CHANGELOG
- [ ] `bench/validate_cmc_*.jl`：G8、G9、G11 を実行して記録する
- [ ] G12 の本番スモーク
- [ ] `CLAUDE.md`（Coupled code sites と Standing contracts）、`.claude/agents/*.md` の更新
- [ ] Tier 2 の 4 軸パネル

## Exit checklist

Run through every item once implementation lands. ~~Strike through~~ items
that do not apply.

- [ ] `make test-all` passes (4 threads).
- [ ] `make docs` builds (strict).
- [ ] If results or a determinism contract changed: test added with an
      implementation-independent oracle; bitwise gates still hold.
- [ ] ~~If `src/gpu/` changed: device validation on a CUDA node recorded in
      `.claude/bench_log.md`.~~
- [ ] If public API changed: `SPEC.md` and `docs/src/api.md` updated;
      dependent-package-facing names flagged.
- [ ] Decision record(s) under `docs/specs/` updated.
- [ ] If a hot path was touched: before / after recorded in
      `.claude/bench_log.md`.
- [ ] Tier 2 review panel run and findings resolved.
- [ ] If module names or Makefile targets changed: `.claude/agents/` swept.
- [ ] `CHANGELOG.md` `[Unreleased]` updated.
- [ ] `Status:` line here and the table in `docs/specs/README.md` updated.
- [ ] Implementation commit hash appended below.
