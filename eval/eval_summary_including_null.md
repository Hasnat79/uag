
# Metrics recall @ 1 iou=m including null values


## evaluation with Video-LLM [unusual activity localization]
| Video-LLM Models | Dataset Name | IoU = 0.001 R@1 | IoU = 0.01 R@1 | IoU = 0.1 R@1 | IoU = 0.3 R@1 | IoU = 0.5 R@1 | IoU = 0.7 R@1 |mIoU|null values|
|------------|--------------|----------------|----------------|---------------|---------------|---------------|---------------|--|--|
| videochat2 | uag_oops_v1 | 37.76 | 37.13 | 28.19 | 13.72 | 4.72 | 0.88 | 10.48 | 124|
| video_chatgpt | uag_oops_v1 | 55.70 | 54.37 | 43.61 | 22.09 | 8.56 | 2.27 | 18.82 |274|
| video_llama2 | uag_oops_v1 | 70.61 | **69.86** | **60.04** |**36.63** | 17.68 | 5.41 | 24.47 |15|
|random| uag_oops_v1| **72.25** | 68.91 | 40.65 | 20.70 |**19.8**| **6.80** | **26.21** |0|
|---|---|---|---|---|---|---|---|---|
| videochat2 | ssbd |10.58 | 10.58 |5.77 | 2.88 | 0.96 | 0.00 |2.12| 0 |
| video_chatgpt | ssbd | 13.46 | 13.46 | 9.62 | 4.81 | 2.88 | 0.00 | 4.11 |22|
| video_llama2 | ssbd |34.62 | 34.62 |29.81 | 15.38| 8.65|1.92 | 11.32 |0|
|random|ssbd|**50**| **54.81**|**41.35**|**14.42**|**5.77**| **4.81**| **13.08**|0|
<!-- |random|ssbd|**50**|**50**|**37.5**|1**6.3**|**8.7**|**2.9**| **13.08**| -->

---
---
## Zeroshot evaluation with LLM models 
| Description Generator| LLM Model |Dataset| IoU = 0.001 R@1 | IoU = 0.01 R@1 | IoU = 0.1 R@1 | IoU = 0.3 R@1 | IoU = 0.5 R@1 | IoU = 0.7 R@1 |mIoU|null values|
|------------|--------------|----------------|----------------|---------------|---------------|---------------|---------------|--|--|--|
|Blip2| llama3 [to be updated] | uag_oops_v1 | 50.98 | 49.97 | 40.91 | 16.99 | 5.54 | 1.64 | 12.93 |8|
|videollama2| llama3| uag_oops_v1 |49.65 | 48.52 | 39.77 | 17.50 | 6.80 | 1.51 | 13.15 |10|
|random prediction |--|uag_oops_v1| **72.25** | 68.91 | 40.65 | 20.70 |**19.8**| **6.80** | **26.21** |0|
|--|--|--|--|--|--|--|--|--|--|--|
|Blip2| llama3 | ssbd | 22.12 | 20.19 |  8.65 | 1.92 | 1.92 | 1.92 |  4.53 |29|
|videollama2| llama3| ssbd | 22.12 | 21.15 | 7.69 | 2.88 | 0.96 | 0.00 | 3.71 |29|
|random|--|ssbd|**50**| **54.81**|**41.35**|**14.42**|**5.77**| **4.81**| **13.08**|0|



---
---
<!-- ## One-shot evaluation with LLM models 
| Description Generator| LLM Model |Dataset| IoU = 0.001 R@1 | IoU = 0.01 R@1 | IoU = 0.1 R@1 | IoU = 0.3 R@1 | IoU = 0.5 R@1 | IoU = 0.7 R@1 |mIoU|null values|
|------------|--------------|----------------|----------------|---------------|---------------|---------------|---------------|--|--|--|
|Blip2| llama3 | uag_oops_v1 | --| --| --| --| --| --| --|0|
|videollama2| llama3| uag_oops_v1 | --| --| --| --| --| --| --|0|
|random prediction |--|uag_oops_v1| **72.25** | 68.91 | 40.65 | 20.70 |**19.8**| **6.80** | **26.21** |0|
|--|--|--|--|--|--|--|--|--|--|--|
|Blip2| llama3 | ssbd | --| --| --| --| --| --| --|0|
|random|--|ssbd|**50**| **54.81**|**41.35**|**14.42**|**5.77**| **4.81**| **13.08**|0| -->


---
---

# Metrics recall@1 abs-distance = m including null values
## evaluation with Video-LLM [unusual activity localization]
| Model Name | Dataset Name | threshold 0.001 R@1 abs dist | threshold 0.01 R@1 abs dist | threshold 0.1 R@1 abs dist | threshold 0.3 R@1 abs dist | threshold 0.5 R@1 abs dist | threshold 0.7 R@1 abs dist | mean Abs dist | null values |
|------------|--------------|------------------------|------------------------|-----------------------|-----------------------|-----------------------|-----------------------|---|---|
| videochat2 | uag_oops_v1  | 11.45                  | 11.64                  | 14.29                 | 21.71                 | 27.63                 | 34.24                 | 237.97| 124 |
| video_chatgpt | uag_oops_v1 | 14.60                  | 14.60                  | 16.93                 | 22.40                 | 27.06                 | 31.09                 |12.63| 274 |
| video_llama2 | uag_oops_v1  | 27.00                  | 27.19                  | 30.02                 | 35.81                 | 40.28                 | 44.37                 |11.40| 15 |
| random | uag_oops_v1| 47.89 | 48.65  |52.61 | 57.21 | 63.37 |67.15 |**5.46**| 0 |
|---|---|---|---|---|---|---|---|---|---|
|videochat2 | ssbd| 8.65 | 8.65 | 10.58 |16.35 | 20.19 |  20.19 | 139.63| 0 |
| video_chatgpt | ssbd         | 9.62                   | 9.62                   | 11.54                 | 15.38                 | 18.27                 | 19.23                 |101.16| 22 |
| video_llama2 | ssbd         |**25.00**                  | **25.00**                  | **26.92**                 | **27.88**                 | **30.77**                 | **30.77**                 |96.55| 0 |
| random | ssbd| 16.35 | 20.19 |18.27 |19.23 | 24.04 |29.81 |75.37| 0 |

---
---
## Zeroshot evaluation with LLM models 
| Description Generator| LLM Model | Dataset Name | threshold 0.001 R@1 abs dist | threshold 0.01 R@1 abs dist | threshold 0.1 R@1 abs dist | threshold 0.3 R@1 abs dist | threshold 0.5 R@1 abs dist | threshold 0.7 R@1 abs dist | mean Abs dist | null values |
|---|------------|--------------|------------------------|------------------------|-----------------------|-----------------------|-----------------------|-----------------------|---|---|
| blip2| Llama3| uag_oops_v1 | 42.48| 42.61| 45.81| 54.50| 62.43| 67.02| 6.08|8|
| videollama2| Llama3| uag_oops_v1 | 48.46| 48.65| 50.66| 58.72| 65.58| 68.97| 5.81|10|
| random prediction| | uag_oops_v1| 47.89 | 48.65  |52.61 | 57.21 | 63.37 |67.15 |**5.46**| 0 |
|---|---|---|---|---|---|---|---|---|---|---|
|blip2| llama3 | ssbd |  23.08                  | 23.08                  |  23.08                 | 25.96                 | 28.85              |  29.81                |66.55| 29 |
|videollama2| llama3 |ssbd | 22.12 |22.12|22.12|24.04|29.81|31.73|**54.19**|29|
| random prediction | | ssbd| 16.35 | 20.19 |18.27 |19.23 | 24.04 |29.81 |75.37| 0 |



