# 预注册：缺陷物理签名 × 光照扰动模式的交互预测

**写定时间**：2026-09-15，**在运行 MVTec 扰动实验之前**。
本文件在看到任何 MVTec 扰动结果前锁定：(a) 缺陷类型的物理签名分类，
(b) 可证伪的预测，(c) 判定规则。事后不得修改；如需修改须另开文件并说明理由。

## 由来

AeBAD-S 上得到的机理主张（见 `refine-logs/EXPERIMENT_RESULTS.md`）：

> 哪一种采集偏移致命，取决于缺陷依赖何种物理签名成像。
> 依赖**明暗**的缺陷（fracture）在光照变化下退化最重，甚至异常分反号；
> 依赖**几何轮廓**的缺陷（breakdown / groove）则对**视角**更敏感。

该主张若为真，应在**另一数据集、另一组缺陷类型**上重现。MVTec AD 有 48 种
命名缺陷、73 个 (类别,缺陷) 组合，且可施加**受控**光照扰动（AeBAD 的真实偏移混杂多因子）。

## (a) 物理签名分类（锁定）

依据：该缺陷的**判别证据**主要来自什么成像量。

### S = shading（明暗型）
细微起伏，其对比度由定向光掠射产生；换光照方向即摧毁签名。
`scratch`, `scratch_head`, `scratch_neck`, `crack`, `poke`, `poke_insulation`,
`fold`, `rough`, `squeeze`, `squeezed_teeth`, `bent`, `bent_wire`, `bent_lead`

### A = albedo（反照率/色彩型）
材质或颜色差异，与光照几何基本无关（但受色温/曝光影响）。
`color`, `contamination`, `metal_contamination`, `oil`, `liquid`, `glue`,
`glue_strip`, `print`, `faulty_imprint`, `gray_stroke`, `thread`,
`thread_side`, `thread_top`

### G = geometry/silhouette（几何轮廓型）
材料缺失/增添/错位，改变形状或版图。
`broken_large`, `broken_small`, `broken`, `broken_teeth`, `split_teeth`, `hole`,
`cut`, `cut_lead`, `cut_inner_insulation`, `cut_outer_insulation`,
`missing_cable`, `missing_wire`, `cable_swap`, `misplaced`, `flip`,
`manipulated_front`, `damaged_case`, `pill_type`, `fabric_border`, `fabric_interior`

### X = 排除（定义不清或混合）
`combined`（多缺陷混合）, `defective`（toothbrush，未细分）

## (b) 可证伪的预测

扰动模式按其改变的成像量分为两组：

| 组 | 模式 | 改变什么 |
|---|---|---|
| **空间光照** | `gradient`, `specular`, `shadow` | 光照的**空间分布** → 破坏明暗签名 |
| **色调/色彩** | `exposure`, `gamma`, `wb` | **全局色调映射** → 破坏反照率签名 |

**P1（主预测）**：在**空间光照**扰动下，S 类缺陷的 AUROC 退化显著大于 G 类。
**P2**：在**色调/色彩**扰动下，A 类缺陷的退化显著大于 G 类。
**P3（交叉）**：「S 在空间光照下的退化 − S 在色调下的退化」显著大于
A 的同一对比量。即**签名类型与扰动类型之间存在交互**，而非某类缺陷普遍更脆弱。

P3 是核心：它把主张从"某些缺陷更脆弱"提升为"**脆弱性由签名与扰动的匹配决定**"。

## (c) 判定规则（锁定）

- 度量：Δ = AUROC(severity=s) − AUROC(severity=0)，逐 (类别,缺陷) 计算。
  severity=0 **严格恒等于原图**（`illum.py` 已自检），构成同一流程内的零扰动对照。
- 统计：以 (类别,缺陷) 为单位做**配对**比较（同一类别下 S/A/G 共享同一参考集与正常图）。
  配对 bootstrap 95% CI（10000 次）+ Mann-Whitney U（跨签名类别为独立样本）。
- **多重比较**：预注册的检验族 = {P1, P2, P3} × {3 个非零 severity} = 9 个检验，
  Holm-Bonferroni 校正。族外的一切比较标注为**探索性**，不作为结论。
- **证伪**：若 P3 的 CI 含 0，则「签名 × 扰动」交互主张在 MVTec 上**不成立**，
  须如实报告，且 AeBAD 上的机理解释降级为该数据集的局部现象。

## (d) 实验参数（锁定）

- 检测器：AnomalyDINO，DINOv2-S/14，448px，1-NN，`mean_top1p`
- 预处理：masking=False, rotation=True（与 AeBAD 主扫描一致）
- 支撑集：每类别取 train/good 的前 k=4 张（8 个 seed，不相交连续块）
- 扰动：**只施加于测试图**（正常与异常图一视同仁），参考图保持标称条件
- severity ∈ {0, 0.33, 0.67, 1.0}；6 种模式
- 每个 (类别, 模式, severity, seed) 评测该类别全部测试图
