# Requirements: constrained Monte Carlo (CMC) for finite-temperature anisotropy

Status: agreed (2026-11-21)

## Goal

全体磁化 M = Σ_s w_{a(s)} e_s の向きを n̂ に固定した拘束アンサンブルを、Asselin 型の 2 スピン移動でサンプルする
（Asselin et al., PRB 82, 054415 (2010)）。そこで平均内部トルク ⟨T⟩ = −n̂ × ∂F/∂n̂ を、全体と項の部分集合ごとに測る。
これにより、有限温度の磁気異方性自由エネルギー F(n̂; T) の角度微分と、その on-site・二イオンなどの寄与を、SCE 模型から直接得られるようにする。

## Background

- **動機。** L1₀ FePt の K(T) を求め、on-site と二イオンの寄与を分けたい（作者の研究用途）。
  SCEMonteCarlo.jl にも、凍結した SpinClusterMC.jl にも CMC は無い。SpinClusterMC の `docs/specs/260512-simple-impl/design.md:76-104` に拡張の設計メモがあるだけ。
- **事前調査。** サブエージェント 3 本が独立に調べた（2026-11-21）。全文は `research/`、試作スクリプトは `research/scripts/`。
  - A 理論：Asselin 2010 を全文読み、ヤコビアンを独立に導出した。N=3 で数値検算した。
  - B 統合：既存コードと契約への影響。
  - C 検証：オラクルとミューテーションの対応表。
- **レビュー（2026-11-21）。** v1 の草案に対して 2 本。
  - spec-review：major 11 / minor 17
  - numerical-review：major 4 / minor 11。導出は独立にやり直して正しいと確認された。指摘の要点は次のとおり。
    - 負の重みのテストが無い
    - on-site と二イオンを分けて測れない
    - `:auto` が弱い拘束を黙って作る
    - G7 が弱い
  - 指摘はすべて v2 に反映した。
- **ユーザーが決めたこと（2026-11-21）。**
  - 第 1 版は全体拘束だけ。拘束付き PT とブロック拘束は v2 以降の別仕様にする。
  - 拘束変数は、原子ごとの符号付き重み w_a の一般形。既定（`:auto`）は `magmoms`（H が持っていれば）、無ければ全部 1。
  - API は新しいドライバ `run_cmc`。既存の `run_mc` / `run_pt` のシグネチャは変えない。
  - 非活性サイトは M から除外する。
  - 項の部分集合ごとのトルク（on-site と二イオンの分離用）を v1 に入れる。
  - `:auto` は残し、弱い拘束（反強磁性・補償点付近）には警告を出す。
- **触れる決定記録。**
  - 新設：`constrained-mc.md`
  - 追記：`binning-observables.md`（トルク系の観測量）、`checkpoint-schema.md`（kind `"cmc"`、C3 の kind 別の記述）、`pt-threads-determinism.md`（P6 に `run_cmc`）
  - `updates-stationarity.md` は変えない。CMC は U1 に乗らないことを `constrained-mc.md` 側に書く。

## Scope

Includes:

- 拘束の値型 `MagnetizationConstraint`（方向 n̂ と、cell atom ごとの符号付き重み）。
- 2 スピン CMC 移動：
  - 主スピンは回転提案だけ。
  - 補償は重み付きで、∥ 成分の符号を保つ。
  - 受理則は Asselin のヤコビアン付き。
  - 幾何的に不可能な提案は null move として数える。
- 重み 0 の活性サイトは、スイープの中で通常の 1 スピン Metropolis で動かす（ハイブリッド CMC）。
- 2 スピン同時のエネルギー差を逐次 leave-one-out で正確に計算する（何体項でも正しい）。
- トルク：
  - 全トルク `total_torque`（Zeeman 込み）
  - CMC の観測量セット（`:torque`、`:M_par`、`:constraint_residual`、磁場があれば `:torque_zeeman`）
  - 項の部分集合ごとのトルク観測量 `torque_observable`
- 拘束付きの初期化（剛体回転）と、決定論的な再初期化（RNG を消費しない）。
- 弱い拘束、ステップの下限張り付き、拘束のずれに対する警告。
- 単一チェーンのドライバ `run_cmc`（温度ラダーと焼きなましは `run_mc` と同じ形）。
- 複数の拘束方向を、独立レーンとしてスレッド並列に回す（`ntasks` によらずビット一致）。
- checkpoint / resume（kind `"cmc"`、1 レーン 1 ファイル）。
- docs：guide（K の取り出し例、on-site と二イオンの分け方と解釈上の注意、重みの選び方）、theory、API、決定記録、CHANGELOG。
- bench（CMC スイープの allocs と時間）と手動検証のスクリプト（G8、G9、G11）。

Excludes（v1 では作らない）:

- 拘束付き PT（同じ拘束どうしなら交換則は今のままで使えることを確認済み。v2 で別仕様にする）。
- ブロック・セル拘束（A(T) や LLB セル用。v2 以降）。サイトごとの重み（磁気セル ≠ 化学セルの Néel ベクトル）。
- |M| の拘束（縦方向の自由エネルギー）。
- CMC での overrelaxation と、M を保つペア共回転の手（v2 の候補）。
- 色クラスごとの並列 CMC スイープ（成り立たない）。
- GPU。
- 拘束付きの `minimize_energy`。
- K(T) や κ_lm のフィットなどの後処理（別パッケージで行う方針）。
- 状態によらない偏ったパートナー規則と、null を除いた適応（O1、O8。v1.1 で検討）。
- SLCEMonteCarlo.jl への移植。

## Invariants

- 既存のドライバ（`run_mc` / `run_pt` / `gpu_run_pt` / `minimize_energy`）の結果は変わらない。
  - mover のリファクタは G15（固定 seed の regression pin、rtol 1e-10）で確かめる。
  - 作業中には、同じ OS・同じ Julia でのビット一致も一度確かめる。
  - `pt.jl` は変えない。
- Spins stay unit vectors; inactive sites stay bitwise frozen and are excluded
  from M, from the pair candidates, and from observables.
- `(4π)^(body/2)` applied once in the `TiledHamiltonian` ctor; `j0` excluded.
  CMC adds no new scaling（部分 Hamiltonian も公開の ctor を通る）。
- `temperature` xor `kT`; β only in accept steps.
- 外部磁場の表現（fitted 項の後ろに付く body-1 項、magmoms ≥ 0）は変えない。
- 既存の更新の定常性（U1–U3）と、serial ≡ parallel のビット一致契約は変わらない（CMC はこれらに乗らない）。
- PT thread-count independence, CPU ≡ GPU: untouched (no change to `pt.jl`,
  `_swap_payload!`, `ChainState`, `src/gpu/`).
- resume ≡ 中断なし（測定途中で中断）は、既存の kind でも `"cmc"` でも成り立つ。
- Checkpoint schema version stays v3（kind の追加は `gpu_pt` の前例に倣う）。既存 kind のファイルは従来どおり読める。
- `_fingerprint` / `model_fingerprint` は変えない。
- 依存パッケージ向けの名前（`energy_gradient!`, `model_fingerprint`, `_gradient_lane_ref!`）は変えない。
- 新しい外部依存は入れない。

## Completion criteria

- [x] `make test-all` passes（4 threads）。新しいゲートは design.md §7.2 の G0–G7、G13–G17。（`1ac5601` で `make test-ci` 26056/26056 + strict docs、着地前の修正後も通過）
- [x] 統計ゲートごとに、オラクルは実装から独立で、σ は余裕つき。分解すべき変異を一時パッチで当てて落ちることを確かめ、シフト量と σ を決定記録に残す。
- [x] 既存ドライバの中立性（G15）が緑。リファクタ時のビット一致の確認も記録する。
- [x] CMC スイープのアロケーションが 0（G16）。M5 の前後の bench を `.claude/bench_log.md` に記録する。
- [x] `make docs` が strict で通る。
- [x] 決定記録 `constrained-mc.md` を新設し、`binning-observables.md`、`checkpoint-schema.md`、`pt-threads-determinism.md`、`docs/specs/README.md` を更新する。（`pt-threads-determinism.md` P6 の 1 行は着地前の修正で追加）
- [x] 手動検証：G8、G9、G11 を 1 回ずつ実行して記録する。（決定記録 `constrained-mc.md` の「Manual validation」、2026-11-21、`c44f764`）
- [x] 本番スモーク G12：FePt または Nd₂Fe₁₄B の l02 模型で、⟨n̂·T⟩ = 0 と、対称性が許す閉路の条件を確かめる。部分トルクの和が全トルクになることも確かめる。（Nd₂Fe₁₄B l02 の再フィット + 合成異方性、seed 2 本、同じ表）
- [x] `CLAUDE.md` と `.claude/agents/` を更新する。
- [x] Tier 2 の 4 軸レビューパネル。

## References

- P. Asselin et al., PRB 82, 054415 (2010), arXiv:1006.3507（受理則：p.7 step 7、式 (A11)。トルク恒等式：App. C、式 (C15)）。
- Y. Toga et al., arXiv:1606.00333（重み付き CMC、Nd₂Fe₁₄B）。
- M. O. A. Ellis et al., arXiv:1706.05188（VAMPIRE のハイブリッド CMC、トルクの直接フィット）。
- Q. Gong et al., arXiv:1902.05636（F = −∫T_y dθ、K1–K3 のフィット）。
- PDF は手元の `references/papers/` にある（gitignore）。各文献のノートは M0 で `references/*.md` に作る。
- SpinClusterMC.jl `docs/specs/260512-simple-impl/design.md:76-104`（旧設計メモ）。
- 事前調査とレビューの検算：`research/A-theory.md`, `research/B-integration.md`, `research/C-oracles.md`, `research/scripts/`。
