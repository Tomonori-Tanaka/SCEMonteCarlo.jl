# Tasklist: constrained Monte Carlo (CMC) for finite-temperature anisotropy

Status: agreed (2026-11-21) — M1/M2 完了、M3 へ

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
- [ ] `MagnetizationConstraint`（内部 ctor、`==` / `isequal` / `hash`）、`_ResolvedConstraint`、重みの解決
- [ ] `_cmc_pair_proposal`（NaN を安全に扱う比較）、拘束付きの初期化
- [ ] 補償スピン e_j′ の |z| ≤ 1 を保証（正規化かクランプ）
- [ ] G4、G0、G1 の幾何の部分（Fmix、F2w±）。変異：ヤコビアン、|z| 比、z 符号、重みの取り違え、|w| 比、実験室座標系での補償、sign(w) の欠落

### M4 — CMC スイープ

- 新しい保証：
  - 定常分布が拘束測度に一致する（G5、G6、G7）。
  - M は不変（G1）。
  - 非活性サイトと重み 0 の規約（G1b）。
  - スイープ中のアロケーションは 0（G16）。
- 維持する保証：既存のスイープには触れない。
- [ ] `_attempt_cmc_pair!`、`cmc_sweep!`（直列、null move を数える、ハイブリッド、毎スイープの M 再計算、警告）
- [ ] `_CMCScratch` に 𝒞 の索引配列を足し、mover 構築時に `length(scratch.lc) == H.nlm` を 1 回 assert
- [ ] 統計ゲートは `cmc_sweep!` と binning を直接回す内部ハーネスで実行する（G5、G6、G7）。G6 は M5 で `run_cmc` 経由でも 1 本足す
- [ ] 統計ゲートの変異を一時パッチで当てて落ちることを確かめ、シフト量と σ を記録する（null move を数えないやり直しを含む）
- [ ] `bench_sweeps.jl` に CMC を加える

### M5 — ドライバ・観測量・API ドキュメント

- 新しい保証：`run_cmc` が、温度ラダー、焼きなまし、拘束付きの初期化と再初期化、弱い拘束の警告を行う。
- 維持する保証：既存ドライバの結果は変わらない（G15）。
- [ ] リファクタ前のコミットで G15 の pin を取る（変化検出器と明記）。`bench_run.jl` と `bench_sweeps.jl` の前を計測する
- [ ] `run.jl` に mover の新メソッドを足す（5 引数の既存メソッドは残す）。作業中に、同じ OS・同じ Julia でビット一致を一度確かめ、ここに記録する
- [ ] `run_cmc`（単一）、`CMCResult`（+ `show`）、`cmc_observables`、`cmc_evaluables`
- [ ] bench の後を計測し、`.claude/bench_log.md` に記録する
- [ ] `binning-observables.md` の B 項、`docs/src/api.md`（export と同じマイルストーンで入れ、strict ビルドを保つ）

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
