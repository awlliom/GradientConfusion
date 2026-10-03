# Angle Analysis for Defense Evaluation

This script analyzes the angular deviations between defense components during adversarial attacks on CIFAR-10 using the Square Attack.

## Overview

A. The script calculates and visualizes the angles between:
1. **A·b*** and **A^T·v** - where b* is the output of `compute_batch_jv_chunked`
2. **A·B_opt** and **A^T·v** - where B_opt is the output of `best_effort_match_sparse`

Where:
- **A**: Jacobian of logits with respect to inputs (derivative matrix)
- **A^T·v**: Gradient of loss with respect to inputs
- **b***: The tensor returned by `compute_batch_jv_chunked` (equals A·A^T·v)
- **B_opt**: The optimized gradient vector from `best_effort_match_sparse`

B. The `loss_gradient_angle_plot_from_json.py` script  compare Angular deviation between defense-exposed and undefended gradient directions over 100 Square Attack iterations on 10 CIFAR-10 samples.
## Requirements

Make sure you have the following dependencies installed:
```bash
pip install torch torchvision numpy scipy matplotlib pandas
```

## How to Run

Execute the script from the project root directory:

```bash
python angle_analysis_cifar10.py
```

For getting the Fig. 2 in paper, execute the following command
```bash
python3 loss_gradient_angle_plot_from_json.py \ 
  --input-json loss_gradient_angle_compare.json \
  --output-dir angles_results \                      
  --interval-size 10
```
One also can get the `loss_gradient_angle_compare.json` by running the following command:

```bash
python3 loss_gradient_angle_cifar10_compare.py \
  --config config-jsons/cifar10_square_linf_config.json \
  --defense-config config-jsons/defense_config.json \
  --num-samples 10 \
  --max-iters 100 \
```

## Configuration

The `angle_analysis_cifar10.py` script is configured to:
- Test on **10 samples** from CIFAR-10
- Run **1000 iterations** of Square Attack per sample
- Use the ResNet model specified in the config
- Calculate angles at each iteration for each sample

You can modify these parameters by editing the config override section in the script (lines 214-215):
```python
config['num_eval_examples'] = 10  # Number of samples
config['attack_config']['max_loss_queries'] = 1000  # Number of iterations
```

## Output

The script creates an `angle_analysis_results/` directory containing:

1. **angles_data.json**: Raw angle measurements for all samples and iterations
   - Format: JSON file with two dictionaries, one for each angle type
   - Each sample has a list of angles (one per iteration)

2. **histogram_A_bstar_vs_ATv.pdf**: Polar histogram showing angular deviations between A·b* and A^T·v
   - Displays distribution of angles across all iterations and samples
   - Includes mean, median, and total count statistics
   - PDF format for easy LaTeX integration

3. **histogram_A_Bopt_vs_ATv.pdf**: Polar histogram showing angular deviations between A·B_opt and A^T·v
   - Same format as above, for the second comparison
   - PDF format for easy LaTeX integration

## Understanding the Plots

The polar histograms show:
- **Angle bins**: 10-degree bins from 0° to 180°
- **Radial axis**: Count of angles in each bin
- **Colors**: Gradient from light to dark blue indicating density
- **Statistics box**: Mean, median, and total number of measurements

The plots are formatted with:
- Dark edges (linewidth 0.5) for reproduction quality
- High resolution (300 DPI)
- Professional styling matching the example figure


## Technical Details

### Angle Calculation Method

For each iteration and each sample, the script:
1. Computes the defense components (b*, B_opt, A^T·v)
2. Calculates A·b* and A·B_opt using automatic differentiation
3. Computes the angle between these vectors and A^T·v using:
   ```
   angle = arccos(dot(normalize(vec1), normalize(vec2))) * 180/π
   ```

### Memory Management

The script uses chunking and gradient cleanup to handle GPU memory efficiently:
- Chunks of size 10 for batch processing
- Automatic CUDA cache clearing
- Detached tensors for storage

## Troubleshooting

**Out of Memory Error:**
- Reduce `chunk_size` parameter in `compute_batch_jv_chunked_with_components` (line 80)
- Reduce number of samples or batch size

**Slow Execution:**
- Ensure CUDA is available: `torch.cuda.is_available()` should return `True`
- Reduce number of iterations or samples for testing

**Import Errors:**
- Make sure you're running from the project root directory
- Check that all project modules are in the Python path

