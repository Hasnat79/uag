## Summary
| Model Name | Dataset Name | Total Videos Predicted | Null Values After Fix | Videos with Extracted Timestamps |
|------------|--------------|-----------------------|----------------------|---------------------------------|
| videochat2 (Video-LLM) | uag oops dataset v1 | 1589 | 124 | 1465 |
| video_chatgpt (Video-LLM) | uag oops dataset v1 | 1589 | 274 | 1315 |
| video_chatgpt (Video-LLM) | ssbd dataset | 104 | 22 | 82 |
| video_llama2 (Video-LLM)| uag oops dataset v1 | 1589 | 15 | 1574 |
| video_llama2 (Video-LLM)| ssbd dataset | 104 | 0 | 104 |
| blip2 (text representation) +llama3 | uag oops dataset v1 (Manual check pending) | 1589 | 77 | 1512 |
| blip2 (text representation)+llama3 | ssbd dataset (Manual check pending) | 104 | 57 | 47 |


## Summary

| Experiment Results Files | Total Videos Predicted | Null Values After Fix | Videos with Extracted Timestamps |
|---------|-----------------------|----------------------|---------------------------------|
| videochat2_pred_uag_oops_dataset_null_fixed.json | 1589 | 124 | 1465 |
| video_chatgpt_pred_uag_oops_dataset_v1_null_fixed.json | 1589 | 274 | 1315 |
| video_chatgpt_pred_ssbd_dataset_null_fixed.json | 104 | 22 | 82 |
| video_llama2_pred_uag_oops_dataset_null_fixed.json | 1589 | 15 | 1574 |
| video_llama2_pred_ssbd_dataset_null_fixed.json | 104 | 0 | 104 |
| (**Manual check pending**) llama3_pred_x_blip2_text_rep_uag_oops_dataset_v1_null_fixed.json | 1589 | 77 | 1512 |
(**Manual check pending**) llama3_pred_x_blip2_text_rep_ssbd_dataset_null_fixed.json | 104 | 57 | 47 |

## videochat2_pred_uag_oops_dataset_null_fixed.json
* final version of videochat2 prediction over uag oops dataset v1
* Total videos predicted for: 1589
* Null values after fix: 124
* Total videos we could extract start_pred and end_pred : 1465
    * we extracted the timestamps using pattern matching and heuristics
        * the patterns are written in fix_null_pred.py
        
* total videos the model did not predict any timestamps: 124
---
## video_chatgpt_pred_uag_oops_dataset_v1_null_fixed.json
* final version of video_chatgpt prediction over uag oops dataset v1
* Total videos predicted for : 1589
* Null values after fix: 274
* Total videos we could extract start_pred and end_pred : 1315
    * we extracted the timestamps using pattern matching and heuristics
        * the patterns are written in fix_null_pred.py
---
## video_chatgpt_pred_ssbd_dataset_null_fixed.json
* final version of video_chatgpt prediction over ssbd dataset
* Total videos predicted for : 104 (actually 58 videos, but total behaviour annotations were 104, so we predicted for all)
* Null values after fix: 22
* Total videos we could extract start_pred and end_pred : 82
    * we extracted the timestamps using pattern matching and heuristics
        * the patterns are written in fix_null_pred.py

## video_llama2_pred_uag_oops_dataset_null_fixed.json
* final version of video_llama2 prediction over uag oops dataset v1
* Total videos predicted for : 1589
* Null values after fix: 15
    * Total null values after first filtration were: 42
    * Then I did manual check and fixed 27 of them
* Total videos we could extract start_pred and end_pred : 1574

## video__llama2_pred_ssbd_dataset_null_fixed.json
* final version of video_llama2 prediction over ssbd dataset
* Total videos predicted for : 104 (actually 58 videos, but total behaviour annotations were 104, so we predicted for all)
* Null values after fix: 0
    * Total null values after first filtration were: 3
    * Then I did manual check and fixed 3 of them
* Total videos we could extract start_pred and end_pred : 104


## llama3_pred_x_blip2_text_rep_uag_oops_dataset_v1_null_fixed.json
* current version of llama3 prediction over uag oops dataset v1 (manual check is pending)
* Total videos predicted for : 1589
* Null values: 77
* Total videos we could extract start_pred and end_pred : 1512

## llama3_pred_x_blip2_text_rep_ssbd_dataset_null_fixed.json
* current version of llama3 prediction over ssbd dataset (manual check is pending)
* Total videos predicted for : 104 (actually 58 videos, but total behaviour annotations were 104, so we predicted for all)  
* Null values: 57
* Total videos we could extract start_pred and end_pred : 47