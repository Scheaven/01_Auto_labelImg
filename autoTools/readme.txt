This is my first purely manual algorithm project!

python trackerManager.py --model_path /data/98_model/03_siamese/tracker_model.pth --video /data/96_data/track_2out.mp4

训练的源模型方法
pysot-master


源项目训练模型的readme
test
python tools/demo.py \
    --config experiments/siamrpn_mobilev2_l234_dwxcorr/config.yaml \
    --snapshot /data/98_model/03_siamese/siamrpn_mobilev2_l234_dwxcorr.pth
    --video /data/96_data/track_2out.mp4 # (in case you don't have webcam)


python tools/demo.py \
    --config experiments/siamrpn_alex_dwxcorr/config.yaml \
    --snapshot /data/01_project/04_tracking/pysot-master/experiments/siamrpn_alex_dwxcorr/snapshot/checkpoint_e19.pth
    --video /data/96_data/track_2out.mp4 # (in case you don't have webcam)
    
training
python -m torch.distributed.launch  --nproc_per_node=1     --master_port=2333     ../../tools/train.py --cfg config.yaml


注意事项：
1、对backbone主干网络的修改，为了和标签匹配，可能需要有一个裁剪操作，以保证他们之间的差值是25
2、如果层数修改了，校验的时候会报错，需要修改一下config.py中校验的层
3、追踪时需要注意self.score_size 需要和输出的tensor的宽高配套
