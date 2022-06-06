# Tutorial code

Note that we have preprocessed CFP images by [CLAHE](https://en.wikipedia.org/wiki/Adaptive_histogram_equalization#Contrast_Limited_AHE) and OCT by a [median filter](https://en.wikipedia.org/wiki/Median_filter), and then resized both to 448x448. So novel test samples are supposed to go through the same preprocessing before feeding them into our models.

+ [count_data.ipynb](count_data.ipynb): Count images / eyes / subjects per dataset, helping users understand how the meta data is organized.
+ [inference-and-eval-single-modal.ipynb](inference-and-eval-single-modal.ipynb): Evaluate a single-modal CNN, *i.e.* CFP-CNN or OCT-CNN, on a given test set.
+ [inference-and-eval-multi-modal.ipynb](inference-and-eval-multi-modal.ipynb): Evaluate a multi-modal CNN (MM-CNN) on a given test set.
+ [notebooks/inference-and-synthesize-fake-images.ipynb](notebooks/inference-and-synthesize-fake-images.ipynb): Run and synthesize fake images.
