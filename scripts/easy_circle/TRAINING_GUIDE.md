# Easy Circle Training Guide

## 🧪 Test Run (CURRENT CONFIG - ~30 minutes)

Your config is currently set for a **quick test run**:

```yaml
max_train_steps: 300  # ~30 minutes
```

### Test Run Details:
- **Duration**: ~30 minutes (300 steps × 6 sec/step)
- **Coverage**: ~8.6% of one epoch (300/3,500 steps)
- **Validation**: Every 50 steps (you'll see 6 validation checkpoints)
- **Checkpoints**: Every 100 steps (3 checkpoints saved)

### Expected Results:
The model will start learning but **won't be fully trained**. This test run will:
- ✅ Verify everything works (cache, training loop, validation)
- ✅ Show you the validation images in TensorBoard
- ✅ Confirm GPU utilization and memory usage
- ✅ Let you see training speed and estimate full training time

---

## 🚀 Full Training Options

After the test run succeeds, edit `configs/easy_circle_flux_kontext.yaml`:

### Option 1: Quick Training (1 epoch, ~6 hours)
```yaml
train:
  # max_train_steps: 300  # Comment out or remove
  num_epochs: 1  # Add this line
```
- Steps: 3,500
- Time: ~6 hours
- Good for: Simple tasks, quick results

### Option 2: Balanced Training (3 epochs, ~17.5 hours) **RECOMMENDED**
```yaml
train:
  # max_train_steps: 300  # Comment out or remove
  num_epochs: 3  # Add this line
```
- Steps: 10,500
- Time: ~17.5 hours
- Good for: Quality results, most tasks

### Option 3: Thorough Training (5 epochs, ~29 hours)
```yaml
train:
  # max_train_steps: 300  # Comment out or remove
  num_epochs: 5  # Add this line
```
- Steps: 17,500
- Time: ~29 hours
- Good for: Best quality, complex patterns

---

## 📐 The Math Explained

### Your Dataset:
- Training samples: 7,000
- Batch size: 2

### Calculations:
```
Steps per epoch = Total samples / Batch size
                = 7,000 / 2
                = 3,500 steps

Time per epoch = 3,500 steps × 6 sec/step
               = 21,000 seconds
               ≈ 5.8 hours

For N epochs:
Total steps = N × 3,500
Total time  = N × 5.8 hours
```

### Examples:
| Epochs | Steps  | Time     | Use Case |
|--------|--------|----------|----------|
| 0.086  | 300    | 30 min   | **Test run (current)** |
| 1      | 3,500  | 5.8 hr   | Quick training |
| 3      | 10,500 | 17.5 hr  | Recommended |
| 5      | 17,500 | 29 hr    | High quality |
| 10     | 35,000 | 58 hr    | Very thorough |

---

## 🎯 Recommended Workflow

### Step 1: Test Run (30 minutes) - **DO THIS FIRST**
```bash
cd /home/ubuntu/sanjana-fs/qwen-image-finetune/scripts/easy_circle
./train_easy_circle.sh
```

**What to check:**
- Does cache build successfully?
- Does training start without errors?
- Can you see validation images in TensorBoard?
- What's the actual training speed (check logs)?

### Step 2: Analyze Test Results

Launch TensorBoard:
```bash
tensorboard --logdir=/home/ubuntu/sanjana-fs/qwen-image-finetune/outputs/easy_circle_lora
```

Look at:
- Training loss (should be decreasing)
- Validation images at steps 50, 100, 150, 200, 250, 300
- GPU memory usage in terminal logs

### Step 3: Decide on Full Training

Based on test results, edit the config and choose:
- **1 epoch** if model learns quickly
- **3 epochs** for balanced quality (recommended)
- **5+ epochs** if you want best quality

### Step 4: Resume for Full Training

The framework will automatically continue from where the test run stopped! Just update the config and run:
```bash
./train_easy_circle.sh
```

Or to be explicit about resuming:
```yaml
resume: /home/ubuntu/sanjana-fs/qwen-image-finetune/outputs/easy_circle_lora/checkpoint-300
```

---

## 💡 Pro Tips

### Monitoring During Test Run
Watch these in the terminal:
```
Step X/300: loss=Y.YYY, lr=Z.ZZZZ
GPU Memory: XX.X GB
Time per step: ~6 seconds
```

### Early Stopping
If validation images look great before training completes:
- Press `Ctrl+C` to stop
- Use the last checkpoint for inference
- No need to train longer!

### Cache Building Time
First time only (~10-15 minutes):
- Processes all 7,000 training samples
- Saves VAE and text encoder outputs
- Makes training 2-3x faster

---

## 🚀 Start Your Test Run Now!

```bash
cd /home/ubuntu/sanjana-fs/qwen-image-finetune/scripts/easy_circle
./train_easy_circle.sh
```

After 30 minutes, check the results and decide how long to train for real! 🎨

