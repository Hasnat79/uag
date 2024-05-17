## Summary

| Experiment Results Files | Total Videos Predicted | Null Values After Fix | Videos with Extracted Timestamps |
|---------|-----------------------|----------------------|---------------------------------|
| videochat2_pred_uag_oops_dataset_null_fixed.json | 1589 | 124 | 1465 |
| video_chatgpt_pred_uag_oops_dataset_v1_null_fixed.json | 1589 | 274 | 1315 |
| video_chatgpt_pred_ssbd_dataset_null_fixed.json | 104 | 22 | 82 |

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