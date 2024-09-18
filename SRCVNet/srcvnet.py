import os
import glob
import time
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.layers import Input, AveragePooling2D, Conv2D
from PIL import Image
from .feature import FeatureExtraction
from .cost import CostConcatenation
from .aggregation import Hourglass, FeatureFusion
from .computation import Estimation, Estimation2
from .refinement import Refinement
from .data_reader import read_left, read_right
from .evaluation import evaluate_all
from .EfficientAttention import EfficientAttention, CBAMBlock
from .costrefinement import cost_refinement


class SRCVNet:
    def __init__(self, height, width, channel, min_disp, max_disp, n_gradients=2):
        self.height = height
        self.width = width
        self.channel = channel
        self.min_disp = int(min_disp)
        self.max_disp = int(max_disp)
        self.n_gradients = n_gradients
        self.model = None
        
    def get_config(self):
        return None
    
    def build_model(self):
        left_image = keras.Input(shape=(self.height, self.width, self.channel))
        right_image = keras.Input(shape=(self.height, self.width, self.channel))
        gx = keras.Input(shape=(self.height, self.width, self.channel))
        gy = keras.Input(shape=(self.height, self.width, self.channel))
        
        #feature extraction
        feature_extraction = FeatureExtraction(filters=16)
        [l0, l1] = feature_extraction(left_image)
        [r0, r1] = feature_extraction(right_image)
        
        #cost function
        cost0 = CostConcatenation(min_disp=self.min_disp // 4, max_disp=self.max_disp // 4)
        cost1 = CostConcatenation(min_disp=self.min_disp // 8, max_disp=self.max_disp // 8)
        
        #4d cost volume
        cost_volume0 = cost0([l0, r0])
        cost_volume1 = cost1([l1, r1])

        self_cost_volume0 = cost0([l0, l0])
        self_cost_volume1 = cost1([l1, l1])
        
        
        #cost refinement
        cost_refiner0 = cost_refinement(filters=32)
        cost_refiner1 = cost_refinement(filters=32)
        cost_refined0 = cost_refiner0([cost_volume0,self_cost_volume0])
        cost_refined1 = cost_refiner1([cost_volume1,self_cost_volume1])

        
        #hourglass module
        hourglass3 = Hourglass(filters=16)
        hourglass2 = Hourglass(filters=16)
        hourglass1 = Hourglass(filters=16)
        
        
        #Efficient atteetion after aggregation
        ea3 = EfficientAttention(in_channels=48, key_channels=48, head_count=8, value_channels=48)
        ea2 = EfficientAttention(in_channels=24, key_channels=24, head_count=8, value_channels=24)
        ea1 = EfficientAttention(in_channels=48, key_channels=48, head_count=8, value_channels=48)
        

        #agg_cost3 = hourglass3(cost_volume0)
        agg_cost3 = hourglass3(cost_refined0)
        #agg_cost3 = hourglass2(attention3)
        attention3 = ea3(agg_cost3)
        
        #agg_cost2 = hourglass2(cost_volume1)
        agg_cost2 = hourglass2(cost_refined1)
        #agg_cost2 = hourglass2(attention2)
        attention2 = ea2(agg_cost2)
        
        estimator2 = Estimation(min_disp=self.min_disp // 8, max_disp=self.max_disp // 8)
        disparity2 = estimator2(agg_cost2)
        
        #multiscale fusion
        fusion2 = FeatureFusion(units=16)
        fusion_cost2 = fusion2([attention2, attention3])
        
        #EA, Hourglass
        agg_fusion_cost1 = hourglass1(fusion_cost2)
        attention1 = ea1(agg_fusion_cost1)
        
        #estimator, disparity
        estimator0 = Estimation(min_disp=self.min_disp // 4, max_disp=self.max_disp // 4)        
        disparity0 = estimator0(attention1)
        
        refiner1 = Refinement(filters=32)
        refined1 = refiner1([disparity0, left_image, gx, gy])
        
        self.model = keras.Model(inputs=[left_image, right_image, gx, gy],
                                 outputs=[disparity2, disparity0, refined1])
        self.model.summary()

    def predict(self, lefts, rights, output_dir, weights):
        self.model.load_weights(weights)
        assert len(lefts) == len(rights)
        t0 = time.time()
        for left, right in zip(lefts, rights):
            left_image, gx, gy = read_left(left)
            left_image = np.expand_dims(left_image, 0)
            gx = np.expand_dims(gx, 0)
            gy = np.expand_dims(gy, 0)
            
            right_image = np.expand_dims(read_right(right), 0)
            disparity = self.model.predict([left_image, right_image, gx, gy])
            disparity = Image.fromarray(disparity[-1][0, :, :, 0])
            
            name = os.path.basename(left) 
            name = name.replace('LEFT_RGB', 'LEFT_DSP')
            disparity.save(os.path.join(output_dir, name))
            
        t1 = time.time()
        print('Total time: ', t1 - t0)

if __name__ == '__main__':
    # # predict
    # left_dir = 'the directory of left images'
    # right_dir = 'the directory of right images'
    # output_dir = 'the directory to save results'
    # weights = 'the weight file'
    
    # net = SRCVNet(1024, 1024, 1, -128.0, 64.0)
    # net.build_model()
    # net.predict(left_dir, right_dir, output_dir, weights)

    # # evaluation
    # gt_dir = 'the directory of ground truth labels'
    # evaluate_all(output_dir, gt_dir, -128.0, 64.0)

    pass
