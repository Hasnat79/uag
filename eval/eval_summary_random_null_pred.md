
# Evaluation Summary (random null pred)
## 1 Metrics recall @ 1 iou=m 


### 1.1 evaluation with Video-LLM [unusual activity localization]
| Video-LLM Models | Dataset Name | IoU = 0.001 R@1 | IoU = 0.01 R@1 | IoU = 0.1 R@1 | IoU = 0.3 R@1 | IoU = 0.5 R@1 | IoU = 0.7 R@1 |mIoU|
|------------|--------------|----------------|----------------|---------------|---------------|---------------|---------------|--|
| videochat2 | uag_oops_v1 |         48.40 | 47.26 | 38.07 | 20.20 | 7.36 | 2.27 | 14.06 |
| video_chatgpt | uag_oops_v1 |      71.87 | 71.37 | 59.91 | 33.92 | 15.10 | 4.53 | 22.62 |
| video_llama 1 v(2) | uag_oops_v1 | 73.25 | 73.19| 64.69 | 41.03 | 21.08 | 6.29 | 27.10 |
|random| uag_oops_v1|                **80.11** | **76.02** | **69.79** | **44.56** | **22.40** | **7.99** | **29.72**|
|---|---|---|---|---|---|---|---|---|
| videochat2 | ssbd |         10.58 | 10.58 | 5.77 | 2.88 | 0.96 | 0.00 | 2.12 | 
| video_chatgpt | ssbd |      21.15 | 25.96 | 17.31 | 10.58 | 6.73 | 2.88 | 4.95 |
| video_llama 1 v(2) | ssbd | 34.62 | 34.62 | 29.81 | **15.38**| **8.65**|1.92 | 11.32 |
|random|ssbd|             **50**| **54.81**|**41.35**|14.42|5.77| **4.81**| **13.08**|
<!-- |random|ssbd|**50**|**50**|**37.5**|1**6.3**|**8.7**|**2.9**| **13.08**| -->

---
---
### 1.2 Zeroshot evaluation with LLM models 
| Description Generator| LLM Model |Dataset| IoU = 0.001 R@1 | IoU = 0.01 R@1 | IoU = 0.1 R@1 | IoU = 0.3 R@1 | IoU = 0.5 R@1 | IoU = 0.7 R@1 |mIoU|
|------------|--------------|----------------|----------------|---------------|---------------|---------------|---------------|--|--|
|Blip2| llama3 | uag_oops_v1 |            60.79 | 59.60 | 48.65 | 19.32 | 7.24 | 2.45 | 15.42 |
|videollama 1 v(2)| llama3| uag_oops_v1 | 55.82 | 55.00 | 45.12 | 19.45 | 8.12 | 2.14 | 14.91 |
|random prediction |--|uag_oops_v1|       **80.11** | **76.02** | **69.79** | **44.56** | **22.40** | **7.99** | **29.72**|
|--|--|--|--|--|--|--|--|--|
|Blip2| llama3 | ssbd | 37.50 | 32.69 | 18.27 | 5.77 | 2.88 | 1.92 | 7.20 |
|videollama 1 v(2)| llama3| ssbd | 33.65 | 30.77 | 14.42 | 6.73 | 1.92 | 0.00 | 5.53 |
|random|--|ssbd|**50**| **54.81**|**41.35**|**14.42**|**5.77**| **4.81**| **13.08**|

---
---

## 2 Metrics recall@1 abs-distance = m seconds 
### 2.1 Evaluation with Video-LLM [unusual activity localization]
|Model Name | Dataset Name | R@1 abs distance within 0 sec |R@1 abs distance within 1 sec| R@1 abs distance within 3 sec| R@1 abs distance within 5 sec | R@1 abs distance within 7 sec| Mean Abs distance (seconds) |
|------------|--------------|------------------------|------------------------|-----------------------|-----------------------|--|--|
| videochat2 | uag_oops_v1 |      0.00| 1.57 | 10.95 | 25.93 | 43.36 | 219.32 |
| video_chatgpt | uag_oops_v1 |     0.00 | 2.14 | 14.10 | 33.04 | 47.95 | 10.97 |
| video_llama 1 v(2) | uag_oops_v1 | **0.06** | 2.01 | 15.04 | 34.30 | 49.09 | 11.19 |
| random prediction | uag_oops_v1|  0.00 | **4.85** | **33.29** | **60.29** | **77.03**| **5.12** |
|---|---|---|---|---|---|---|---|
| videochat2 | ssbd |        0.00| 0.00|1.92 | 2.88 | 4.81 | 139.63 |
| video_chatgpt | ssbd |      0.00 |0.00| 0.96 | 2.88 | 4.81 |  99.30 |
| video_llama 1 v(2) | ssbd | 0.00 | 0.00 | **3.85** | **6.73** | **13.46** | 96.55 |
| random prediction | ssbd|   0.00 | **0.96** |0.96| 1.92 |4.81 | 75.68 |



### 2.2 Zeroshot evaluation with LLM models
| Description Generator| LLM Model | Dataset Name | R@1 abs distance within 0 sec |R@1 abs distance within 1 sec| R@1 abs distance within 3 sec| R@1 abs distance within 5 sec | R@1 abs distance within 7 sec| Mean Abs distance (seconds) |
|---|------------|--------------|------------------------|------------------------|-----------------------|-----------------------|--|--|
| blip2| Llama3| uag_oops_v1 |            0.00 | 1.45 |27.00| 53.93 | 73.69 | 5.83 |
| videollama 1 v(2)| Llama3| uag_oops_v1 | 0.00 | 2.01 | 26.31 | 55.25 | 74.20 | 5.71 |
| random prediction| | uag_oops_v1|         0.00 | **4.85** | **33.29** | **60.29** | **77.03** | **5.12** |
|---|---|---|---|---|---|---|---|---|
|blip2| llama3 | ssbd             |0.00|0.00|0.96|3.85|6.73|**75.45**|
|videollama 1 v(2)| llama3 |ssbd | 0.00 | 0.00| 0.96|**4.81**|**7.69**|77.23|
| random prediction | | ssbd      | 0.00 | **0.96** |**0.96**| 2.88 | 4.81 | 75.68 |

<!-- ### 2.1 evaluation with Video-LLM [unusual activity localization]
| Model Name | Dataset Name | threshold 0.001 R@1 abs dist | threshold 0.01 R@1 abs dist | threshold 0.1 R@1 abs dist | threshold 0.3 R@1 abs dist | threshold 0.5 R@1 abs dist | threshold 0.7 R@1 abs dist | mean Abs dist |  
|------------|--------------|------------------------|------------------------|-----------------------|-----------------------|-----------------------|-----------------------|---|
| videochat2 | uag_oops_v1  |19.32 | 19.32 | 23.16 | 31.53 | 41.54 | 49.15 |219.35|
| video_chatgpt | uag_oops_v1 | 28.07 | 28.32 | 30.65 | 38.33 | 44.75 | 50.28 |10.98|
| video_llama 1 v(2) | uag_oops_v1  | 31.78 | 32.35 | 35.31 | 41.66 | 47.07 | 51.48 |11.19|
| random | uag_oops_v1| 55.70 | 56.70  |61.17 | 66.14 | 71.49 |74.83 |**5.18**|
|---|---|---|---|---|---|---|---|---|---|
|videochat2 | ssbd| 8.65 | 8.65 | 10.58 |16.35 | 20.19 | 20.19 | 139.63|
| video_chatgpt | ssbd | 11.54 | 12.50 | 15.38 | 19.23 | 23.08 | 25.00 |102.47|
| video_llama 1 v(2) | ssbd | 25.00 | 25.00 | 26.92 | 27.88 | 30.77 | 30.77 | 96.55|
| random | ssbd| 16.35 | 20.19 |18.27 |19.23 | 24.04 |29.81 |75.37| 0 |

---
---
### 2.2 Zeroshot evaluation with LLM models 
| Description Generator| LLM Model | Dataset Name | threshold 0.001 R@1 abs dist | threshold 0.01 R@1 abs dist | threshold 0.1 R@1 abs dist | threshold 0.3 R@1 abs dist | threshold 0.5 R@1 abs dist | threshold 0.7 R@1 abs dist | mean Abs dist |  
|---|------------|--------------|------------------------|------------------------|-----------------------|-----------------------|-----------------------|-----------------------|--|
| blip2| Llama3| uag_oops_v1 |       52.36| 52.99| 57.58| 66.27| 73.19| 78.16| 5.83|
| videollama2| Llama3| uag_oops_v1 | 56.51| 56.95| 60.92| 67.53| 73.44| 77.28| 5.71|
| random | uag_oops_v1|             |47.89| 48.65  |**52.61** | 57.21 | 63.37 |67.15 |**5.46**|
|---|---|---|---|---|---|---|---|---|---|---|
|blip2| llama3 | ssbd | **28.85**| 23.08 | 25.96 | **29.81** | 31.73 | **38.46**|**72.33**|
|videollama2| llama3 |ssbd | 27.88 | **26.92** | **27.88** | **29.81**| **34.62** | 36.54 |77.90|
| random prediction | | ssbd| 16.35 | 20.19 |18.27 |19.23 | 24.04 |29.81 |75.37| -->

 <!-- Accuracy Within 1 sec| Accuracy Within 0.25 sec| -->
## 3 Unusual Activity Detection Accuracy Comparison 
### 
|Dataset| Model / Method| Accuracy within 1 sec | Accuracy within 0.25 sec |
|---|---|---|---|
|uag_oops_v1| videochat2 |                                              12.59| 5.22|
|uag_oops_v1| video_chatgpt |                                           15.48| 5.98|
|uag_oops_v1| video_llama 1(v2)             |                            20.39| 8.37|
|uag_oops_v1| blip2 + llama3 (our approach)                           | **47.64**| **17.81**|
|uag_oops_v1| videollama 1 (v2) + llama3 (our approach) |              **48.21**| **17.87**|
|uag_oops_v1| random prediction |35.62| 13.78|
|uag_oops_v1|Human Consistency [human prediction found on oops dataset]| 89.43| 56.83|
|---|---|---|---|
|ssbd| videochat2 [Video LLM]|                                6.73| 0.00|
|ssbd| video_chatgpt [Video LLM] |                              9.62| 1.92|
|ssbd| video_llama 1(v2) [Video LLM]|                        **15.38**| **2.88**|
|ssbd| blip2 + llama3 (our approach) |             6.73| 0.00|
|ssbd| videollama 1 (v2) + llama3 (our approach) | **10.58**| **2.88**|
|ssbd| random prediction | 5.77| 1.92|
<!-- |oops_dataset(full)| Video Speed [oops paper]| 65.3|36.6| -->






