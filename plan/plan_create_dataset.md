# PLAN — Synthetic Dataset for Concept Recoverability in Conditional Diffusion

## 0. Mục tiêu

Mục tiêu của dataset là tạo một môi trường **nhẹ, controllable và dễ phân tích** để nghiên cứu câu hỏi:

> Khi nào một concept có thể được cải thiện chỉ bằng cách thay đổi initial noise / seed, và khi nào việc tiếp tục seed search hoặc seed optimization không còn tác dụng vì model không có đủ capability?

Dataset không nhằm mô phỏng đầy đủ text-to-image thực tế. Nó được thiết kế như một **scientific instrument** để isolate các yếu tố:

- Concept identity
- Training support
- Intra-concept diversity
- Initial noise / seed
- Model capability

Mục tiêu chính là phân biệt:

1. **Seed-recoverable failure**
   - Một số initial noises thất bại.
   - Nhưng tồn tại initial noise khác sinh đúng concept.

2. **Capability-limited failure**
   - Thay đổi nhiều initial noises vẫn không tạo đúng concept.
   - Failure có khả năng đến từ learned model distribution, không còn chỉ do chọn sai trajectory.

---

# 1. Research Question

Với concept `c` và conditional diffusion model `G`:

```text
x = G(z0, c)
z0 ~ N(0, I)
```

định nghĩa concept success probability:

```text
q_c = P[G(z0, c) satisfies concept c]
```

Ta muốn nghiên cứu mối quan hệ giữa `q_c` và training support của concept `N_c`.

Interpretation:

- `q_c` lớn: concept dễ sinh.
- `0 < q_c << 1`: concept khó nhưng có thể recover bằng seed search.
- `q_c ≈ 0`: candidate capability-limited concept.

---

# 2. Scope của Dataset V0

Dataset V0 chỉ sử dụng:

- Single object
- One color attribute
- One shape attribute
- Resolution: `32×32`
- RGB images
- Simple fixed background

Không đưa vào V0:

- Multiple objects
- Spatial relation
- Attribute binding giữa nhiều objects
- Text encoder
- CLIP/T5
- Natural backgrounds
- Photorealistic textures
- Occlusion
- Complicated lighting

Lý do: dataset V0 cần tối giản để mọi failure có thể interpret được.

---

# 3. Concept Space

## 3.1 Colors

Sử dụng 6 colors:

```python
COLORS = {
    "red":     (230, 30, 30),
    "green":   (30, 200, 60),
    "blue":    (30, 80, 230),
    "yellow":  (230, 220, 40),
    "cyan":    (40, 210, 210),
    "magenta": (210, 50, 200),
}
```

Yêu cầu:

- Các color phải dễ phân biệt.
- Không được quá gần nhau trong RGB space.
- Color jitter không được làm thay đổi semantic color class.

## 3.2 Shapes

Sử dụng 6 shapes:

```text
circle
square
triangle
diamond
cross
star
```

Yêu cầu:

- Shape phải visually distinguishable ở 32×32.
- Không dùng shape quá phức tạp ở V0.
- Mỗi shape phải có renderer deterministic từ parameters.

## 3.3 Concept Definition

Một concept:

```text
concept = (color, shape)
```

Tổng:

```text
6 × 6 = 36 concepts
```

Ví dụ:

```text
red_circle
red_square
blue_triangle
cyan_star
magenta_cross
```

---

# 4. Concept Support Design

## 4.1 Support Levels

Training support của concept:

```text
N_c ∈ {0, 1, 5, 20, 100, 500}
```

| Support | Meaning |
|---:|---|
| 0 | unseen composition |
| 1 | extremely rare |
| 5 | very rare |
| 20 | rare |
| 100 | moderate support |
| 500 | high support |

Mỗi support level có 6 concepts.

Tổng training samples:

```text
6 × (0 + 1 + 5 + 20 + 100 + 500) = 3756
```

---

# 5. Balanced Support Assignment

Không được assign support theo color hoặc shape cố định.

Ví dụ sai:

```text
red concepts -> high support
blue concepts -> low support
```

Điều này tạo confound giữa `support` và `color`.

## 5.1 Latin-Square Assignment

| Color / Shape | Circle | Square | Triangle | Diamond | Cross | Star |
|---|---:|---:|---:|---:|---:|---:|
| Red | 0 | 1 | 5 | 20 | 100 | 500 |
| Green | 500 | 0 | 1 | 5 | 20 | 100 |
| Blue | 100 | 500 | 0 | 1 | 5 | 20 |
| Yellow | 20 | 100 | 500 | 0 | 1 | 5 |
| Cyan | 5 | 20 | 100 | 500 | 0 | 1 |
| Magenta | 1 | 5 | 20 | 100 | 500 | 0 |

Properties:

- Mỗi color xuất hiện đúng 1 lần ở mỗi support level.
- Mỗi shape xuất hiện đúng 1 lần ở mỗi support level.
- Mỗi support level chứa 6 concepts.
- Support không bị confound trực tiếp với color hoặc shape.

---

# 6. Intra-Concept Diversity

Hai samples cùng concept, ví dụ `red_circle`, phải giữ cùng semantic identity nhưng khác appearance.

Ta định nghĩa:

```text
x = R(c, eta)
```

trong đó:

```text
c   = semantic concept
eta = nuisance / instance parameters
```

## 6.1 Instance Parameters

Mỗi sample có các parameters:

```text
center_x
center_y
size
rotation
color_jitter
brightness_jitter
renderer_seed
```

Optional ở các version sau:

```text
edge_softness
small deformation
```

## 6.2 Suggested Ranges

### Position

```text
center_x ∈ [10, 22]
center_y ∈ [10, 22]
```

Yêu cầu:

- Object luôn nằm trong canvas.
- Không crop object.

### Size

```text
size ∈ [6, 10]
```

Ý nghĩa `size` phụ thuộc shape:

- circle -> radius
- square -> half-width
- triangle -> bounding radius
- star -> outer radius

### Rotation

```text
circle   -> 0
square   -> [-15°, +15°]
triangle -> [-15°, +15°]
diamond  -> [-10°, +10°]
cross    -> [-10°, +10°]
star     -> [-15°, +15°]
```

### Color Jitter

```text
RGB channel jitter ∈ [-10, +10]
brightness scale ∈ [0.9, 1.1]
```

Yêu cầu: jitter không được làm thay đổi semantic color class.

---

# 7. Diversity Constraint

Dataset không chỉ cần nhiều samples mà còn cần tránh near-duplicates.

Với mỗi sample:

```text
eta_i = (x_i, y_i, size_i, rotation_i, rgb_i)
```

có thể định nghĩa normalized distance:

```text
D(i,j) = w_p D_position + w_s D_size + w_r D_rotation + w_c D_color
```

Sample mới được accept nếu:

```text
min_j D(i,j) > tau
```

Không nhất thiết check với toàn bộ dataset nếu support=500; có thể check nearest recent samples hoặc dùng KD-tree.

---

# 8. Diversity không được confound với Support

Ta muốn thay đổi `N_c` nhưng giữ underlying instance distribution giống nhau:

```text
eta ~ p(eta)
```

cho tất cả concepts.

Do đó `support=5` và `support=500` phải sampling từ cùng parameter distribution.

Không được thiết kế:

```text
support=500 -> vị trí, size, rotation rất đa dạng
support=5   -> gần như identical
```

Nếu không, ta không biết failure do low support hay low diversity.

Lưu ý: empirical diversity của support nhỏ tự nhiên không thể bằng support lớn. Điều cần kiểm soát là **sampling distribution giống nhau**, không phải sample variance bằng nhau tuyệt đối.

---

# 9. Dataset Generation Strategy

## 9.1 Master Pool

Khuyến nghị tạo trước một master pool lớn:

```text
1000 instances / concept
```

Tổng:

```text
36 × 1000 = 36,000 images
```

Master pool dùng để:

- chọn generator training samples
- tạo evaluator dataset
- tạo test samples
- visualize distribution

## 9.2 Generator Training Split

Từ master pool, lấy số sample theo support:

```text
0, 1, 5, 20, 100, 500
```

Mỗi concept dùng support level tương ứng Latin-square.

Ví dụ `red_circle` có support=0:

- không có sample `red_circle` trong generator training set
- `red` vẫn xuất hiện với các shapes khác
- `circle` vẫn xuất hiện với các colors khác

Điều này bắt buộc để support=0 biểu diễn **unseen composition**, không phải unseen primitive.

---

# 10. Test Dataset

Test dataset phải độc lập với generator training images.

Khuyến nghị:

```text
200 test samples / concept
```

Tổng:

```text
36 × 200 = 7200 test images
```

Test set bao gồm tất cả 36 concepts, kể cả support=0.

Test samples phải:

- cùng renderer distribution
- khác renderer seed
- không overlap metadata với train
- không duplicate image

---

# 11. Evaluator Dataset

Nếu dùng learned evaluator:

```text
Evaluator training:
36 concepts × 500 samples = 18,000 images
```

Evaluator dataset phải balanced giữa concepts.

Không dùng generator training support distribution cho evaluator.

Evaluator phải nhận diện tốt tất cả 36 concepts, kể cả concept support=0 đối với generator.

---

# 12. Metadata Schema

Mỗi image phải có metadata.

Khuyến nghị CSV hoặc Parquet:

```text
sample_id
split
concept_id
color_name
color_id
shape_name
shape_id
support_level
center_x
center_y
size
rotation
base_rgb_r
base_rgb_g
base_rgb_b
final_rgb_r
final_rgb_g
final_rgb_b
brightness
renderer_seed
image_path
```

Ví dụ:

```text
sample_id       = red_circle_000023
split           = master
concept_id      = red_circle
color_name      = red
color_id        = 0
shape_name      = circle
shape_id        = 0
support_level   = 0
center_x        = 17
center_y        = 14
size            = 8.3
rotation        = 0
renderer_seed   = 91831
image_path      = master/images/red_circle/red_circle_000023.png
```

---

# 13. Folder Structure

```text
dataset/
├── configs/
│   ├── concepts.yaml
│   ├── support_matrix.yaml
│   └── renderer.yaml
├── master/
│   ├── images/
│   └── metadata.parquet
├── generator_train/
│   ├── images/
│   └── metadata.parquet
├── generator_test/
│   ├── images/
│   └── metadata.parquet
├── evaluator_train/
│   ├── images/
│   └── metadata.parquet
├── evaluator_val/
│   ├── images/
│   └── metadata.parquet
└── visualizations/
```

---

# 14. Suggested Code Structure

```text
src/
├── dataset/
│   ├── shapes.py
│   ├── colors.py
│   ├── renderer.py
│   ├── sampler.py
│   ├── support_assignment.py
│   ├── generate_master_pool.py
│   ├── build_generator_split.py
│   ├── build_evaluator_split.py
│   ├── validate_dataset.py
│   └── visualize_dataset.py
└── utils/
    ├── seed.py
    └── io.py
```

---

# 15. Renderer API

Khuyến nghị interface:

```python
def render_sample(color_name, shape_name, renderer_seed, config):
    # returns image, metadata
    ...
```

Renderer phải deterministic:

```python
render_sample("red", "circle", renderer_seed=123, config=config)
```

luôn trả cùng image và metadata.

---

# 16. Shape Renderer

Implement từng primitive độc lập:

```python
draw_circle(...)
draw_square(...)
draw_triangle(...)
draw_diamond(...)
draw_cross(...)
draw_star(...)
```

Mỗi function nhận:

```text
center
size
rotation
color
```

Khuyến nghị:

- dùng PIL.ImageDraw hoặc OpenCV
- nếu aliasing mạnh, render ở resolution cao hơn rồi downsample
- mọi shape phải dùng cùng rendering strategy

---

# 17. Randomness Management

Không dùng global random state tùy ý.

Mỗi sample có `renderer_seed` riêng:

```python
rng = np.random.default_rng(renderer_seed)
```

Mọi random parameter của sample lấy từ RNG này.

Không phụ thuộc vào global `np.random.seed(...)` giữa workers.

---

# 18. Split Reproducibility

Tạo global dataset seed:

```text
DATASET_SEED = 20260914
```

Seed này dùng để:

- select master-pool indices cho splits
- shuffle dataset
- optional support assignment nếu sau này randomize

Mỗi image vẫn có `renderer_seed` riêng.

---

# 19. Dataset Validation

Trước khi train diffusion, dataset phải pass validation.

## 19.1 Count Validation

Check:

```text
36 concepts
6 support levels
6 concepts/support
```

Training count:

```text
support=0   -> 0 × 6
support=1   -> 1 × 6
support=5   -> 5 × 6
support=20  -> 20 × 6
support=100 -> 100 × 6
support=500 -> 500 × 6
```

Expected total:

```text
3756
```

## 19.2 Primitive Coverage

Đối với mỗi support=0 concept, kiểm tra cả hai primitives vẫn xuất hiện ở training set.

Ví dụ `red_circle` support=0:

```text
red_square, red_triangle, ... phải tồn tại
blue_circle, green_circle, ... phải tồn tại
```

---

# 20. Train/Test Leakage Check

Verify theo từng concept:

```text
renderer_seed(train) ∩ renderer_seed(test) = empty
```

Ngoài ra check image hash:

```text
image_hash(train) ∩ image_hash(test) = empty
```

---

# 21. Diversity Validation

Cho mỗi concept tính:

```text
std(center_x)
std(center_y)
std(size)
std(rotation)
std(R)
std(G)
std(B)
```

Visualize parameter distributions giữa concepts và support groups.

Không yêu cầu variance bằng nhau tuyệt đối; chỉ yêu cầu cùng renderer distribution.

---

# 22. Visual Sanity Check

Generate contact sheet cho 36 concepts.

Mỗi concept visualize khoảng:

```text
16–25 instances
```

Tạo:

```text
visualizations/all_concepts_grid.png
```

Ngoài ra tạo diversity grid cho một số concepts:

```text
visualizations/red_circle_diversity.png
visualizations/blue_square_diversity.png
visualizations/cyan_star_diversity.png
```

Checklist bằng mắt:

- concept đúng màu
- concept đúng shape
- không crop
- không quá nhỏ
- không quá lớn
- variation đủ nhìn thấy
- semantic identity không thay đổi

---

# 23. Quantitative Diversity Check

Đối với concept có đủ samples, report:

```text
mean pairwise parameter distance
median pairwise parameter distance
minimum pairwise parameter distance
```

Nếu minimum distance gần 0, kiểm tra near-duplicate.

---

# 24. Pixel Duplicate Check

Hash mỗi image, ví dụ SHA-256, và verify không có duplicate ngoài trường hợp cố ý.

Không dùng pixel MSE như metric diversity duy nhất vì translation nhỏ có thể tạo MSE lớn dù semantic giống nhau.

---

# 25. Semantic Correctness Validation

Vì renderer synthetic, tận dụng ground-truth renderer metadata.

Color validation có thể dùng:

- foreground mask
- mean RGB
- nearest predefined color

Shape validation có thể dùng:

- renderer mask
- contour checks
- optional small classifier

---

# 26. Dataset Versioning

Bắt đầu:

```text
Shapes32-Recoverability-v0.1
```

Mỗi lần thay đổi:

- color palette
- renderer ranges
- support matrix
- shapes
- split logic

phải bump version.

---

# 27. Config Example

`renderer.yaml`

```yaml
image_size: 32

background:
  rgb: [0, 0, 0]

position:
  min_x: 10
  max_x: 22
  min_y: 10
  max_y: 22

size:
  min: 6
  max: 10

rotation:
  square: [-15, 15]
  triangle: [-15, 15]
  diamond: [-10, 10]
  cross: [-10, 10]
  star: [-15, 15]
  circle: [0, 0]

color_jitter:
  enabled: true
  delta: 10

brightness:
  min: 0.9
  max: 1.1
```

---

# 28. Implementation Order

## Phase 1 — Concept specification

- [ ] Define 6 colors
- [ ] Define 6 shapes
- [ ] Define 36 concept IDs
- [ ] Define support levels
- [ ] Save Latin-square support matrix
- [ ] Fix dataset seed

Output:

```text
concepts.yaml
support_matrix.yaml
```

## Phase 2 — Renderer

- [ ] Implement blank canvas
- [ ] Implement each shape
- [ ] Implement position sampling
- [ ] Implement size sampling
- [ ] Implement rotation
- [ ] Implement color jitter
- [ ] Implement metadata return
- [ ] Make renderer deterministic

Output:

```text
renderer.py
```

## Phase 3 — Renderer validation

- [ ] Generate 100 samples/concept
- [ ] Make concept grid
- [ ] Inspect boundaries
- [ ] Check all shapes recognizable
- [ ] Check all colors recognizable
- [ ] Adjust rendering ranges

**Do not continue if renderer is not visually clean.**

## Phase 4 — Master pool

- [ ] Generate 1000 samples/concept
- [ ] Save images
- [ ] Save metadata
- [ ] Compute hashes
- [ ] Validate no duplicates
- [ ] Validate parameter ranges

Output:

```text
36,000 master images
master_metadata.parquet
```

## Phase 5 — Generator training split

- [ ] Apply Latin-square support assignment
- [ ] Sample correct number of instances/concept
- [ ] Build `generator_train`
- [ ] Check total = 3756
- [ ] Validate atomic primitive coverage

Output:

```text
generator_train/
generator_train_metadata.parquet
```

## Phase 6 — Generator test split

- [ ] Sample 200 unseen instances/concept
- [ ] Ensure no training overlap
- [ ] Ensure all 36 concepts represented equally
- [ ] Hash-check leakage

Output:

```text
generator_test/
7200 images
```

## Phase 7 — Evaluator split

- [ ] Balanced all 36 concepts
- [ ] 500 train images/concept
- [ ] Separate validation set
- [ ] No overlap with generator test
- [ ] Prefer no overlap with generator train

Output:

```text
evaluator_train/
evaluator_val/
```

## Phase 8 — Final dataset audit

- [ ] Counts correct
- [ ] Support assignment correct
- [ ] Primitive coverage correct
- [ ] No train/test duplicate
- [ ] Metadata complete
- [ ] Renderer configs stored
- [ ] Visual grid acceptable
- [ ] Dataset reproducible from seed

---

# 29. Dataset Ready Criteria

Dataset chỉ được xem là ready khi:

## Structure

- [ ] 36 concepts tồn tại.
- [ ] Support levels đúng.
- [ ] Latin-square assignment đúng.

## Semantic validity

- [ ] >99% images visually represent intended concept.
- [ ] Color classes distinguishable.
- [ ] Shape classes distinguishable.

## Diversity

- [ ] Same concept có observable instance variation.
- [ ] Không có systematic duplicates.
- [ ] Same renderer distribution cho mọi support level.

## Reproducibility

- [ ] Renderer deterministic.
- [ ] Dataset generation script reproducible.
- [ ] Config + seeds được lưu.

## Leakage

- [ ] Generator train và test không overlap.
- [ ] Evaluator không phụ thuộc support imbalance của generator.

---

# 30. Minimal First Milestone

Trước khi tạo toàn bộ dataset, tạo prototype:

```text
2 colors:
red
blue

2 shapes:
circle
square
```

Tổng:

```text
4 concepts
```

Generate:

```text
100 samples / concept
```

Visualize trước khi scale lên 36 concepts.

Mục đích: tránh phát hiện lỗi renderer sau khi đã generate hàng chục nghìn images.

---

# 31. Experiment Extension — Không làm ngay

## V1 — Two-object binding

Ví dụ:

```text
red circle + blue square
```

Research question: model có bind đúng color với đúng object không?

## V2 — Spatial relation

Ví dụ:

```text
red circle left of blue square
```

## V3 — Controlled rare primitives

Tạo rare atomic shape hoặc rare color.

Setting này phải tách khỏi V0 vì câu hỏi scientific khác unseen composition.

---

# 32. Main Scientific Risks

## Risk 1 — Support=0 vẫn generalize quá dễ

Nếu conditional embedding và rendering quá đơn giản, model có thể generalize gần hoàn hảo sang unseen compositions.

Điều này không có nghĩa experiment sai; nó cho thấy V0 quá dễ để xuất hiện capability boundary.

Next step có thể là:

- tăng composition complexity
- thêm texture
- thêm two-object binding
- tăng nonlinear interaction giữa attributes

## Risk 2 — Model không học được support=500

Nếu high-support concepts vẫn fail, chưa được kết luận capability-limited.

Kiểm tra:

- model capacity
- convergence
- conditioning implementation
- diffusion schedule
- sampling procedure

## Risk 3 — Evaluator bias

Nếu evaluator đúng trên rendered images nhưng fail trên generated images, metric có distribution-shift problem.

Mitigation:

- visual audit
- rule-based color detection
- multiple evaluation methods
- evaluator calibration

## Risk 4 — Number of samples và empirical diversity bị confound

Support lớn tự nhiên cover instance space rộng hơn support nhỏ.

Paper phải phát biểu chính xác:

> Mọi concept được sampled từ cùng underlying renderer distribution; training support điều khiển số observations từ distribution đó.

Không claim empirical diversity giống nhau giữa support levels.

---

# 33. Recommended Immediate Next Tasks

Thứ tự thực hiện:

1. Implement `concepts.yaml`
2. Implement `support_matrix.yaml`
3. Implement shape renderer
4. Generate prototype 4 concepts
5. Visualize sample grids
6. Tune position / size / rotation ranges
7. Expand to 36 concepts
8. Generate master pool
9. Build generator train/test splits
10. Run dataset validation

Chỉ sau khi 10 bước trên pass mới bắt đầu conditional diffusion training.

---

# 34. Expected Deliverables

Cuối giai đoạn dataset cần có:

```text
configs/
    concepts.yaml
    renderer.yaml
    support_matrix.yaml

src/dataset/
    renderer.py
    generate_master_pool.py
    build_generator_split.py
    build_evaluator_split.py
    validate_dataset.py
    visualize_dataset.py

dataset/
    master/
    generator_train/
    generator_test/
    evaluator_train/
    evaluator_val/

reports/
    dataset_statistics.json
    dataset_validation.md

visualizations/
    all_concepts_grid.png
    support_matrix.png
    diversity_examples/
```

---

# 35. Definition of Done

Dataset phase hoàn tất khi có thể chạy:

```bash
python -m src.dataset.generate_master_pool
python -m src.dataset.build_generator_split
python -m src.dataset.build_evaluator_split
python -m src.dataset.validate_dataset
```

và validation cuối trả:

```text
[PASS] Concept definitions
[PASS] Support assignment
[PASS] Dataset counts
[PASS] Primitive coverage
[PASS] Train/test independence
[PASS] Metadata integrity
[PASS] Duplicate detection
[PASS] Renderer parameter ranges
[PASS] Visual sanity checks
```

Sau đó project mới chuyển sang conditional diffusion model setup và nghiên cứu:

```text
training support N_c
        ↓
concept success q_c
        ↓
seed-search recovery R_c(K)
```

để xác định seed-recoverable và capability-limited concepts.
