len of failed_validation_videos_with_single_scene: 1589
## uag_oops_dataset_v1.json
- Total videos: 1589
- this dataset contains oops validation filtered validation videos which have only one scene
    - Normally oops videos had multiple scenes in a single video
    - we filtered those videos out and kept only the videos that had a single scene

## blip2_text_rep_x_uag_oops_dataset_v1.json
- Total videos: 1589
- This dataset contains text representation of the videos in uag_oops_dataset_v1.json
    - the text representations were generated using blip2 ~ blip2-opt-2.7b model from salesforce research (total time took: 3:07:43)
- How: 
    - we extracted frames from the video at 1 frame per second rate.
    - we fed the frames of each video to the blip2 model and got the caption of each frame. then we concatenated the captions of all the frames like following example:  
    ```
     "0.0s: a man is standing on the stairs in a living room with a couch in front of him and a phone in his hand.
     1.0s: a living room with a couch and a stairway leading up to the second floor of the house in front of it.
     2.0s: a man is climbing up the stairs in a living room with a couch in the middle of the room next to the stairs.
     3.0s: a man standing on the top of a staircase in a living room with a couch in the middle of the living room.
     4.0s: a man is standing on the stairs in a living room with a couch in front of him and a white wall behind him.
     5.0s: a truck is driving down the road on a foggy day in the middle of a field with grass and trees in the background.
     6.0s: a living room with a couch and a stairway leading up to the second floor of the house in front of it.
     7.0s: a man is standing on the top of a stairway in a living room with a couch in the middle of the room."
    ```