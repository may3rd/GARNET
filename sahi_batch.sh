sahi predict --source PTTEP/ER --name PTTEP_ER --model_type ultralytics \
--model_path yolo_weights/yolov8n_PTTEP_running.pt \
--model_config_path datasets/data/pttep.yaml \
--dataset_json_path coco_images.json --model_device mps --slice_width 640 --slice_height 640 \
--overlap_height_ratio 0.5 --overlap_width_ratio 0.5 --model_confidence_threshold 0.80 \
--visual_bbox_thickness 5 --visual_text_size 0.5 --visual_text_thickness 2 --visual_hide_labels true \
--visual_hide_conf true