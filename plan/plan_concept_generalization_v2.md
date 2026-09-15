# PLAN V2 — Concept Generalization for Shape × Color Recoverability

## 0. Mục tiêu

Mục tiêu của phase này là xây dựng một controlled synthetic benchmark để nghiên cứu:

> Khi direct training exposure của một concept giảm dần, khả năng generate concept đó thay đổi như thế nào, và ở mức exposure nào failure vẫn còn có thể được cứu bằng một training-free method?

Phase này chỉ tập trung vào:

- **Concept Generalization**
- Single object
- Concept được định nghĩa là **Shape × Color**
- Controlled conditional diffusion
- Training-free recovery sau khi model đã được train

Không nghiên cứu trong phase này:

- Multi-object generation
- Spatial relation
- Counting
- Natural-language prompt
- Attribute binding giữa nhiều objects
- Scene complexity cao

---

# 1. Research Question

Định nghĩa một concept:

```text
c = (shape, color)
```

Ví dụ:

```text
(red, circle)
(blue, square)
(yellow, star)
```

Với conditional diffusion model:

```text
x = G_theta(z_T, c)
z_T ~ N(0, I)
```

Ta nghiên cứu hai câu hỏi chính.

### RQ1 — Concept learning under controlled exposure

> Khi exposure của một Shape × Color concept trong training giảm dần, xác suất model generate đúng concept thay đổi như thế nào?

Định nghĩa:

```text
q_rand(c, e)
=
P_z [ G_theta_e(z, c) satisfies c ]
```

trong đó:

- `c`: target Shape × Color concept
- `e`: training exposure level
- `theta_e`: model được train dưới exposure level `e`

### RQ2 — Training-free recoverability

> Khi random generation bắt đầu fail, một fixed training-free method X còn có thể recover concept đó đến mức exposure nào?

Định nghĩa:

```text
q_X(c, e)
=
P [ X(G_theta_e, c) succeeds ]
```

Recovery gain:

```text
Delta_X(c,e)
=
q_X(c,e) - q_rand(c,e)
```

---

# 2. Concept Space

## 2.1 Colors

Dùng 6 colors:

```text
red
green
blue
yellow
cyan
magenta
```

## 2.2 Shapes

Dùng 6 shapes:

```text
circle
square
triangle
diamond
cross
star
```

## 2.3 Concept Definition

Một concept là:

```text
concept = (shape, color)
```

Tổng:

```text
6 shapes × 6 colors = 36 concepts
```

Ví dụ:

```text
circle_red
square_blue
triangle_green
star_yellow
```

Điểm quan trọng:

> Trong phase này, `(shape, color)` được coi là một **single semantic concept**.

Primitive shape và primitive color vẫn được theo dõi riêng để bảo đảm việc giảm exposure của một pair không vô tình biến thành unseen primitive.

---

# 3. Experimental Principle

Kế thừa MOSAIC ở nguyên tắc:

> **Chỉ thay đổi một training-data factor tại một thời điểm; giữ các yếu tố còn lại cố định.**

Các yếu tố giữ cố định:

- Image resolution
- Renderer
- Shape geometry
- Color palette
- Position distribution
- Size distribution
- Rotation policy
- Background
- Conditional encoding
- Diffusion architecture
- Optimizer
- Learning rate
- Training steps
- Total training dataset size
- Evaluator
- Sampling procedure
- Training-free inference budget

Biến chính được manipulate:

```text
target concept exposure
```

---

# 4. Exposure Levels

Thay vì support tuyệt đối:

```text
{0, 1, 5, 20, 100, 500}
```

phase này dùng **relative exposure levels**:

```text
e ∈ {100%, 75%, 50%, 25%, 5%, 0%}
```

Trong đó:

```text
100% = full-support reference
0%   = exact Shape × Color pair không xuất hiện trong training
```

Ví dụ target:

```text
c* = (red, circle)
```

Nếu full-support reference là:

```text
N_full = 500
```

thì:

| Exposure | # red-circle samples |
|---:|---:|
| 100% | 500 |
| 75% | 375 |
| 50% | 250 |
| 25% | 125 |
| 5% | 25 |
| 0% | 0 |

Các con số tuyệt đối có thể thay đổi sau pilot; phần scientific quan trọng là **relative exposure**.

---

# 5. Không dùng Latin-Square Support Matrix làm main experiment

Plan cũ assign mỗi concept một support level khác nhau trong cùng một training dataset.

Ví dụ:

```text
red_circle      -> 0
red_square      -> 1
red_triangle    -> 5
...
```

Cách đó hữu ích cho screening nhưng không sạch cho causal comparison vì:

```text
concept identity
```

bị gắn cố định với:

```text
support level
```

Trong V2, mỗi exposure level phải là một **separate training regime**.

Ví dụ:

```text
Dataset E100
Dataset E75
Dataset E50
Dataset E25
Dataset E05
Dataset E00
```

Mỗi dataset dùng cùng concept space và cùng total N_train.

---

# 6. Target Concept Selection

Không nên chỉ chọn một concept như `red_circle`.

Khuyến nghị chọn một balanced target set.

Ví dụ 6 targets:

```text
red_circle
green_square
blue_triangle
yellow_diamond
cyan_cross
magenta_star
```

Properties:

- mỗi color xuất hiện đúng một lần
- mỗi shape xuất hiện đúng một lần
- không confound exposure với một color/shape cụ thể

Có thể mở rộng thành nhiều balanced target sets nếu compute cho phép.

---

# 7. Primitive Coverage Constraint

Ở exposure = 0%, exact pair không xuất hiện:

```text
(red, circle) -> 0 samples
```

nhưng primitive `red` và primitive `circle` vẫn phải xuất hiện nhiều lần thông qua các pair khác:

```text
red_square
red_triangle
red_star
...

blue_circle
green_circle
yellow_circle
...
```

Do đó:

```text
(red, circle) unseen
```

không đồng nghĩa với:

```text
red unseen
circle unseen
```

Validation bắt buộc:

```text
for each target concept c=(shape,color):
    count(color in other concepts) > threshold
    count(shape in other concepts) > threshold
```

---

# 8. Constant Total Training Budget

Đây là control quan trọng nhất.

Nếu giảm target exposure mà total dataset cũng giảm, ta sẽ confound:

```text
concept exposure
```

với:

```text
overall dataset size
```

Do đó:

```text
N_train(E100)
=
N_train(E75)
=
N_train(E50)
=
N_train(E25)
=
N_train(E05)
=
N_train(E00)
```

Samples bị remove khỏi target concepts phải được redistribute sang non-target concepts.

Redistribution phải:

- balanced theo color
- balanced theo shape
- reproducible
- không tạo một new dominant concept quá mạnh

---

# 9. Suggested Dataset Size

Pilot recommendation:

```text
36 concepts
N_full per concept = 500
```

Balanced reference dataset:

```text
36 × 500 = 18,000 images
```

Nếu 6 target concepts đều được giảm exposure đồng thời:

```text
removed samples
=
6 × (500 - N_target(e))
```

Số sample này được redistribute sang 30 non-target concepts.

Giữ:

```text
total N_train = 18,000
```

cho mọi exposure regimes.

Nếu compute quá lớn, pilot nhỏ hơn:

```text
N_full = 100
total N_train = 3,600
```

sau đó scale lên.

---

# 10. Dataset Regimes

Tạo 6 dataset configs:

```text
concept_generalization_E100.yaml
concept_generalization_E075.yaml
concept_generalization_E050.yaml
concept_generalization_E025.yaml
concept_generalization_E005.yaml
concept_generalization_E000.yaml
```

Mỗi config lưu:

```text
exposure_level
target_concepts
target_count
non_target_redistribution
total_train_size
dataset_seed
renderer_config
```

---

# 11. Renderer — Base Setting

Concept Generalization V1 cần scene đơn giản nhất có thể.

## Fixed

```text
image_size  = 32×32
object_count = 1
background  = fixed
```

## Controlled nuisance variation

Position:

```text
center_x ∈ [10,22]
center_y ∈ [10,22]
```

Size:

Khuyến nghị ban đầu:

```text
fixed size
```

Rotation:

Khuyến nghị ban đầu:

```text
fixed rotation
```

Color:

```text
fixed semantic RGB per class
```

Có thể có rất nhỏ anti-alias/rendering variation, nhưng chưa cần color jitter ở baseline.

### Lý do

Ở experiment đầu tiên, câu hỏi phải là:

```text
Does Shape × Color exposure affect learnability and recoverability?
```

Không nên đồng thời hỏi:

```text
Does size / rotation / color variation affect learnability?
```

Các nuisance factors sẽ được thêm trong complexity extension sau.

---

# 12. Intra-Concept Diversity

Ngay cả khi size và rotation fixed, cùng một concept vẫn cần nhiều instances khác nhau.

Base source of diversity:

```text
position
sub-pixel rendering / anti-alias variation
optional small geometric jitter
```

Nếu chỉ random position chưa đủ, thêm một nuisance variable từng bước.

Nguyên tắc:

```text
eta ~ p(eta)
```

phải giống nhau cho mọi concept và mọi exposure regime.

---

# 13. Master Pool

Tạo master pool độc lập với exposure assignment.

Quy mô đã chốt cho baseline:

```text
500 instances / concept
36 concepts
= 18,000 images
```

Master pool dùng cho:

- generator train selection
- generator test
- evaluator train
- evaluator validation
- sanity check

Master pool không encode exposure level.

Exposure chỉ xuất hiện khi build từng training regime.

---

# 14. Training Split Construction

Pseudo logic:

```text
for exposure in [1.0, .75, .50, .25, .05, 0.0]:

    target_count = round(N_full * exposure)

    for target concept:
        sample target_count instances

    removed_budget = reference_target_budget - selected_target_budget

    redistribute removed_budget to non-target concepts

    assert total_N_train == constant
```

Mỗi regime phải có separate metadata:

```text
generator_train_E100.parquet
generator_train_E075.parquet
generator_train_E050.parquet
generator_train_E025.parquet
generator_train_E005.parquet
generator_train_E000.parquet
```

---

# 15. Test Dataset

Test set phải hoàn toàn balanced.

Khuyến nghị:

```text
200 test instances / concept
36 concepts
= 7,200 images
```

Test set giống nhau cho tất cả models.

Điều này cực kỳ quan trọng:

```text
Model_E100
Model_E75
...
Model_E00
```

đều evaluate trên cùng một test distribution.

---

# 16. Evaluator

Evaluator không được inherit generator exposure imbalance.

Evaluator dataset:

```text
balanced over all 36 Shape × Color concepts
```

Có thể dùng:

1. Rule-based shape/color evaluator
2. Learned classifier
3. Cả hai để cross-check

Với toy renderer, ưu tiên rule-based / geometry-aware evaluator nếu đủ robust.

Success:

```text
success =
shape_correct AND color_correct
```

---

# 17. Conditional Diffusion Training

Train một independent model cho mỗi exposure regime:

```text
Model_E100
Model_E075
Model_E050
Model_E025
Model_E005
Model_E000
```

Tất cả dùng:

```text
same architecture
same optimizer
same learning rate
same number of updates
same diffusion schedule
same conditional encoding
```

Khuyến nghị thêm:

```text
3 training seeds / exposure
```

để tránh conclusion phụ thuộc initialization của model.

Tổng baseline models:

```text
6 exposure levels × 3 training seeds = 18 models
```

---

# 18. Baseline Evaluation — Random Seed Accessibility

Với mỗi model và mỗi target concept:

```text
sample K random diffusion seeds
```

Khuyến nghị pilot:

```text
K = 100
```

Estimate:

```text
q_rand(c,e)
=
# successful random seeds / K
```

Report:

```text
mean ± std over training seeds
```

---

# 19. Training-Free Recovery

Chọn một fixed method X.

Ví dụ phase đầu:

```text
Best-of-N seed search
```

vì đơn giản và dễ interpret.

Sau đó có thể thêm:

```text
initial-noise optimization
```

Giữ inference budget cố định giữa mọi exposure levels.

Ví dụ:

```text
N = 10, 50, 100
```

Recovery metric:

```text
R_X(c,e)
```

và:

```text
Delta_X(c,e)
=
R_X(c,e) - q_rand(c,e)
```

---

# 20. Main Curves

Experiment phải tạo ít nhất 3 curves.

## Curve 1 — Learning curve

```text
Exposure
↓
Random Success
```

Plot:

```text
x-axis: exposure %
y-axis: q_rand
```

## Curve 2 — Recoverability curve

```text
Exposure
↓
Training-Free Success
```

Plot:

```text
x-axis: exposure %
y-axis: R_X
```

## Curve 3 — Recovery gain

```text
x-axis: exposure %
y-axis: Delta_X
```

Mục tiêu là tìm vùng mà:

```text
q_rand low
but
R_X high
```

Đây là candidate **seed-recoverable regime**.

---

# 21. Operational Regions

Không claim capability trực tiếp chỉ từ one method.

Dùng terminology:

### Learned / Easily Accessible

```text
q_rand high
```

### Seed-Recoverable

```text
q_rand low
R_X substantially higher
```

### Weakly Recoverable

```text
q_rand low
R_X improves but remains low
```

### Candidate Capability-Limited

```text
q_rand ≈ 0
R_X ≈ 0
```

Lưu ý:

> Failure của method X không chứng minh mathematically rằng không tồn tại successful seed.

Chỉ kết luận:

```text
unrecoverable under method X and tested inference budget
```

---

# 22. Recoverability Boundary

Có thể operationalize:

```text
e*_X
=
minimum exposure e
such that
R_X(e) >= tau_success
```

Ví dụ:

```text
tau_success = 0.8
```

Nếu:

```text
R_X(50%) = 0.85
R_X(25%) = 0.60
```

thì recoverability boundary nằm giữa:

```text
25% and 50%
```

Nên report interval, không cần ép một threshold chính xác nếu exposure grid còn thưa.

---

# 23. Concept-Level vs Aggregate Analysis

Không chỉ average 6 targets.

Report cả:

```text
per-concept curve
```

và:

```text
aggregate curve across concepts
```

Model:

```text
q(c,e)
```

cho phép kiểm tra:

- một số Shape × Color pair có inherently harder không
- exposure effect có nhất quán không
- color/shape có residual bias không

---

# 24. Required Validation

## Dataset controls

- [ ] 36 Shape × Color concepts
- [ ] 6 balanced target concepts
- [ ] Same total N_train across exposure levels
- [ ] Primitive shape coverage preserved
- [ ] Primitive color coverage preserved
- [ ] Same renderer distribution
- [ ] Same test set
- [ ] No train/test overlap

## Model controls

- [ ] Same architecture
- [ ] Same training steps
- [ ] Same optimizer
- [ ] Multiple training seeds

## Inference controls

- [ ] Same random-seed pool when possible
- [ ] Same training-free method
- [ ] Same search/optimization budget
- [ ] Same evaluator

---

# 25. Folder Structure

```text
dataset/
├── master/
├── test/
├── evaluator/
└── concept_generalization/
    ├── E100/
    ├── E075/
    ├── E050/
    ├── E025/
    ├── E005/
    └── E000/
```

Configs:

```text
configs/
├── concepts.yaml
├── renderer_base.yaml
├── target_concepts.yaml
├── concept_generalization_E100.yaml
├── concept_generalization_E075.yaml
├── concept_generalization_E050.yaml
├── concept_generalization_E025.yaml
├── concept_generalization_E005.yaml
└── concept_generalization_E000.yaml
```

---

# 26. Code Changes from Existing Plan

Giữ lại:

```text
renderer.py
generate_master_pool.py
build_evaluator_split.py
validate_dataset.py
visualize_dataset.py
```

Thay:

```text
support_assignment.py
```

bằng:

```text
exposure_assignment.py
```

Thay:

```text
build_generator_split.py
```

bằng:

```text
build_concept_generalization_splits.py
```

Add:

```text
validate_exposure_regimes.py
```

---

# 27. Implementation Order

## Phase A — Reuse current renderer

- [x] 6 colors
- [x] 6 shapes
- [x] 36 concepts
- [x] Renderer implemented
- [x] Renderer visually validated

## Phase B — Simplify Base renderer

Before final master pool:

- [x] Decide fixed size (`8.0 px` radius/half-width according to shape semantics)
- [x] Decide fixed rotation (`0 degrees` canonical orientation)
- [x] Keep random position only
- [x] Disable unnecessary color/brightness jitter for baseline
- [x] Regenerate sanity grid
- [x] Confirm all concepts visually separable

Frozen Base profile: `concept_generalization_base_c0` in
`dataset/configs/renderer_base.yaml`. The within-concept diversity threshold is
`0.04` over normalized position distance; this rejects near-identical positions
while remaining feasible when position is the only varying nuisance factor.
The frozen validation set contains `500 samples/concept` (`18,000` images total).

## Phase C — Master pool

- [x] Generate master pool (`500 samples/concept`, `18,000` images)
- [x] Hash check
- [x] Save metadata (CSV + Parquet)
- [x] Validate renderer distribution

The master pool is exposure-independent, uses a dedicated deterministic seed
namespace, and stores frozen concept/renderer config snapshots plus a checksum
manifest. Exposure/support labels are intentionally absent from master metadata.

## Phase D — Exposure regimes

- [x] Define balanced target concepts
- [x] Set `N_full = 100`
- [x] Build E100
- [x] Build E075
- [x] Build E050
- [x] Build E025
- [x] Build E005
- [x] Build E000
- [x] Keep total `N_train = 3,600` constant
- [x] Validate redistribution
- [x] Validate primitive coverage

The six regimes are deterministic, nested metadata-only views of the master
pool. The six targets cover every color and shape exactly once. Removed target
samples are distributed equally across all 30 non-target concepts, preserving
exact color and shape marginals of `600` samples each in every regime.

## Phase E — Evaluator

- [x] Build balanced evaluator dataset
- [ ] Train evaluator on Colab (dual-head CNN and end-to-end pipeline implemented)
- [ ] Validate learned evaluator >99% on synthetic held-out test
- [x] Rule-based cross-check: 100% joint accuracy on held-out renderer test
- [ ] Inspect generated-image robustness later

Evaluator allocation from the 500-sample master pool is leakage-free and
balanced: `280 train + 50 validation + 50 test` samples per concept, after a
reserved 120-sample generator prefix. Colab execution instructions are in
`docs/run_evaluator_on_colab.md`.

## Phase F — Diffusion baseline

- [ ] Train E100 model
- [ ] Verify all target concepts learnable
- [ ] Only then train remaining exposure regimes
- [ ] Use multiple training seeds

## Phase G — Random accessibility

- [ ] Generate K random seeds / target concept / model
- [ ] Compute q_rand
- [ ] Plot exposure → random success

## Phase H — Training-free method

- [ ] Implement fixed X
- [ ] Fix inference budget
- [ ] Evaluate all exposure regimes
- [ ] Compute R_X
- [ ] Compute Delta_X
- [ ] Estimate recoverability boundary

---

# 28. Stop Conditions

Không chạy full experiment nếu:

### E100 fails

Nếu high-exposure targets không được generate reliably:

```text
q_rand(E100) low
```

thì chưa được interpret low-exposure failure.

Debug:

- model capacity
- conditioning
- convergence
- diffusion schedule
- evaluator

### E000 trivially succeeds

Nếu:

```text
q_rand(E000) ~ q_rand(E100)
```

thì Shape × Color toy setup quá compositional/easy để tạo boundary.

Next extensions:

- nuisance complexity
- harder shapes
- multi-object binding
- texture
- spatial relations

Nhưng đây là scientific result, không phải implementation failure.

---

# 29. Complexity Extension — Sau Concept Generalization

Chỉ làm sau khi Base exposure curve đã rõ.

Possible controlled variants:

```text
C0: fixed size + fixed rotation + random position
C1: random size
C2: random size + rotation
C3: random size + rotation + distractors
```

Sau đó hỏi:

> Recoverability boundary có dịch chuyển khi scene complexity tăng không?

Không trộn experiment này vào Concept Generalization baseline.

---

# 30. Main Expected Figure

Figure chính nên là:

```text
Success
1.0 |                    Random
    |                ____ Training-free X
    |            ___/
    |        ___/
    |    ___/
0.0 +----------------------------
      0   5  25  50  75  100
           Exposure (%)
```

Vẽ:

- Random generation curve
- Training-free X curve
- confidence interval over target concepts / training seeds

Khoảng cách giữa hai curves:

```text
recoverability gain
```

---

# 31. Scientific Interpretation

Desired chain:

```text
Shape × Color exposure
        ↓
Learned generation accessibility
        ↓
Random-seed success
        ↓
Training-free recoverability
        ↓
Candidate recoverability boundary
```

Điều experiment **có thể support**:

> Training exposure controls how accessible a concept is under random sampling, and training-free seed intervention can recover some low-accessibility concepts up to a certain regime.

Điều experiment **chưa được phép claim trực tiếp**:

> Below exposure X the model has absolutely no capability.

Thay vào đó:

> Below exposure X, the concept was not recoverable under the tested training-free method and inference budget.

---

# 32. Definition of Done — Concept Generalization Phase

Phase hoàn thành khi có:

- [ ] 6 exposure-controlled training datasets
- [ ] Constant total training size
- [ ] Balanced target concepts
- [ ] Primitive coverage preserved
- [ ] Same balanced test set
- [ ] At least 3 training seeds / exposure if compute allows
- [ ] Random success curve
- [ ] Training-free success curve
- [ ] Recovery-gain curve
- [ ] Per-concept results
- [ ] Aggregate results
- [ ] Exposure/recoverability interpretation
- [ ] Clear separation between:
  - random sampling failure
  - recoverable failure
  - unrecoverable-under-X failure

---

# 33. Immediate Next Tasks

1. Freeze Base renderer policy:
   - fixed size
   - fixed rotation
   - random safe position
   - fixed semantic colors

2. Define 6 balanced target concepts.

3. Choose pilot `N_full`:
   - recommended first run: `N_full = 100`
   - scale later to `500`

4. Replace Latin-square split builder with exposure-regime builder.

5. Build only:
   - E100
   - E025
   - E000

   for a first smoke test.

6. Train 3 pilot models.

7. Verify:

```text
q_rand(E100) > q_rand(E025) > q_rand(E000)
```

or document if the trend does not appear.

8. Only after pilot works, expand to:

```text
100%, 75%, 50%, 25%, 5%, 0%
```

9. Add training-free Best-of-N baseline.

10. Plot the first exposure-recoverability curve.
