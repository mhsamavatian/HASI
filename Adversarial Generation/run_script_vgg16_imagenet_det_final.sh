#shuold set cuda to cuda9
python main.py --dataset_name ImageNet --model_name VGG16 --nb_examples 100 \
--attacks "carlinil0?batch_size=1&targeted=next&confidence=5;\
carlinil0?batch_size=1&targeted=ll&confidence=5;\
carlinil2?max_iterations=1000&batch_size=50&targeted=next&confidence=10;\
carlinil2?max_iterations=1000&batch_size=50&targeted=next&confidence=20;\
carlinil2?max_iterations=1000&batch_size=50&targeted=next&confidence=30;\
carlinil2?max_iterations=1000&batch_size=50&targeted=next&confidence=40;\
carlinil2?max_iterations=1000&batch_size=50&targeted=next&confidence=70;\
carlinil2?max_iterations=1000&batch_size=50&targeted=ll&confidence=10;\
carlinil2?max_iterations=1000&batch_size=50&targeted=ll&confidence=20;\
carlinil2?max_iterations=1000&batch_size=50&targeted=ll&confidence=30;\
carlinil2?max_iterations=1000&batch_size=50&targeted=ll&confidence=40;\
carlinil2?max_iterations=1000&batch_size=50&targeted=ll&confidence=70;\
carlinili?batch_size=1&targeted=next&confidence=5;\
carlinili?batch_size=1&targeted=ll&confidence=5;\
eadl1?batch_size=20&max_iterations=1000&targeted=next&confidence=10&beta=0.001;\
eadl1?batch_size=20&max_iterations=1000&targeted=next&confidence=20&beta=0.001;\
eadl1?batch_size=20&max_iterations=1000&targeted=next&confidence=30&beta=0.001;\
eadl1?batch_size=20&max_iterations=1000&targeted=next&confidence=40&beta=0.001;\
eadl1?batch_size=20&max_iterations=1000&targeted=next&confidence=70&beta=0.001;\
eadl1?batch_size=20&max_iterations=1000&targeted=ll&confidence=10&beta=0.001;\
eadl1?batch_size=20&max_iterations=1000&targeted=ll&confidence=20&beta=0.001;\
eadl1?batch_size=20&max_iterations=1000&targeted=ll&confidence=30&beta=0.001;\
eadl1?batch_size=20&max_iterations=1000&targeted=ll&confidence=40&beta=0.001;\
eadl1?batch_size=20&max_iterations=1000&targeted=ll&confidence=70&beta=0.001;\
eaden?batch_size=20&max_iterations=1000&confidence=10&targeted=next&beta=1e-2&abort_early=False;\
eaden?batch_size=20&max_iterations=1000&confidence=10&targeted=ll&beta=1e-2&abort_early=False;" --detection "FeatureSqueezing?squeezers=bit_depth_5,median_filter_2_2,non_local_means_color_11_3_4&distance_measure=l1&fpr=0.05"