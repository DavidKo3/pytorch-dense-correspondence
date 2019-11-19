import torch.nn.functional as F
from torch.autograd import Variable

class DistributionalLoss(object):
	
    def __init__(self, image_width, image_height):
	self.type = "distributional_loss"
	self.image_width = image_width
	self.image_height = image_height

    @staticmethod
    def distributional_loss_single_match(image_a_pred, image_b_pred, match_b, match_a):
        match_b_descriptor = torch.index_select(image_b_pred, 1, match_b) # get descriptor for image_b at match_b 
        p_a = F.softmax(-1 * ((image_a_pred - match_b_descriptor).norm(2, 2).pow(2)), dim=1)
        q_a = gauss_2d(match_a)
    
        print(p_a.shape, p_a.sum())
        #print(descriptor_diffs.shape, descriptor_diffs)
        #p_a = nn.Softmax2d((image_a_pred - match_b_descriptor).pow(2))
        #print(p_a.shape)
        #print(p_a)

    @staticmethod
    def get_distributional_loss(image_a_pred, image_b_pred, matches_a, matches_b):
        print(matches_b.shape)
        distributional_loss_single_match(image_a_pred, image_b_pred, matches_b[0])
