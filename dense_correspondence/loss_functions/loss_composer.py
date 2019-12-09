from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset, SpartanDatasetDataType
from dense_correspondence.loss_functions.pixelwise_contrastive_loss import PixelwiseContrastiveLoss

import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

def get_loss(pixelwise_contrastive_loss, match_type, 
              image_a_pred, image_b_pred,
              matches_a,     matches_b,
              masked_non_matches_a, masked_non_matches_b,
              background_non_matches_a, background_non_matches_b,
              blind_non_matches_a, blind_non_matches_b):
    """
    This function serves the purpose of:
    - parsing the different types of SpartanDatasetDataType...
    - parsing different types of matches / non matches..
    - into different pixelwise contrastive loss functions

    :return args: loss, match_loss, masked_non_match_loss, \
                background_non_match_loss, blind_non_match_loss
    :rtypes: each pytorch Variables

    """
    if (match_type == SpartanDatasetDataType.SINGLE_OBJECT_WITHIN_SCENE).all():
        print "applying SINGLE_OBJECT_WITHIN_SCENE loss"
        return get_within_scene_loss(pixelwise_contrastive_loss, image_a_pred, image_b_pred,
                                            matches_a,    matches_b,
                                            masked_non_matches_a, masked_non_matches_b,
                                            background_non_matches_a, background_non_matches_b,
                                            blind_non_matches_a, blind_non_matches_b)

    if (match_type == SpartanDatasetDataType.SINGLE_OBJECT_ACROSS_SCENE).all():
        print "applying SINGLE_OBJECT_ACROSS_SCENE loss"
        return get_same_object_across_scene_loss(pixelwise_contrastive_loss, image_a_pred, image_b_pred,
                                            blind_non_matches_a, blind_non_matches_b)

    if (match_type == SpartanDatasetDataType.DIFFERENT_OBJECT).all():
        print "applying DIFFERENT_OBJECT loss"
        return get_different_object_loss(pixelwise_contrastive_loss, image_a_pred, image_b_pred,
                                            blind_non_matches_a, blind_non_matches_b)


    if (match_type == SpartanDatasetDataType.MULTI_OBJECT).all():
        print "applying MULTI_OBJECT loss"
        return get_within_scene_loss(pixelwise_contrastive_loss, image_a_pred, image_b_pred,
                                            matches_a,    matches_b,
                                            masked_non_matches_a, masked_non_matches_b,
                                            background_non_matches_a, background_non_matches_b,
                                            blind_non_matches_a, blind_non_matches_b)

    if (match_type == SpartanDatasetDataType.SYNTHETIC_MULTI_OBJECT).all():
        print "applying SYNTHETIC_MULTI_OBJECT loss"
        return get_within_scene_loss(pixelwise_contrastive_loss, image_a_pred, image_b_pred,
                                            matches_a,    matches_b,
                                            masked_non_matches_a, masked_non_matches_b,
                                            background_non_matches_a, background_non_matches_b,
                                            blind_non_matches_a, blind_non_matches_b)

    else:
        raise ValueError("Should only have above scenes?")

def flattened_mask_indices(img_mask, width=640, height=480, inverse=False):
    mask = img_mask.view(width*height,1).squeeze(1)
    if inverse:
    	inv_mask = 1 - mask
    	inv_mask_indices_flat = torch.nonzero(inv_mask)
        return inv_mask_indices_flat
    else:
	return torch.nonzero(mask)

def normalize(x):
    return F.normalize(x, p=1)

def gauss_2d_batch(width, height, sigma, U, V, masked_indices=None, normalize=False):
    U.unsqueeze_(1).unsqueeze_(2)
    V.unsqueeze_(1).unsqueeze_(2)
    X,Y = torch.meshgrid([torch.arange(0., width), torch.arange(0., height)])
    X,Y = torch.transpose(X, 0, 1).cuda(), torch.transpose(Y, 0, 1).cuda()
    G=torch.exp(-((X-U.float())**2+(Y-V.float())**2)/(2.0*sigma**2))
    G=G.view(G.shape[0], G.shape[1]*G.shape[2])
    if masked_indices is not None:
        G[:, masked_indices] = 0.0
    if normalize:
        return normalize(G).double()
    return G.double()

def bimodal_gauss(G1, G2, normalize=False):
    bimodal = torch.max(G1, G2)
    if normalize:
        return normalize(bimodal)
    return bimodal

def distributional_loss_batch(image_a_pred, image_b_pred, matches_a, matches_b, sigma=2, masked_indices=None, symm_matches_a=None, image_width=640, image_height=480):
    num_matches = matches_b.shape[0]
    matches_b_descriptor = torch.index_select(image_b_pred, 1, matches_b)
    matches_b_descriptor = matches_b_descriptor.view(matches_b_descriptor.shape[1], 1, matches_b_descriptor.shape[2])
    norm_degree = 2
    image_a_pred_batch = image_a_pred.squeeze().repeat(matches_b.shape[0], 1).view(matches_b.shape[0], image_a_pred.shape[1], image_a_pred.shape[2])
    descriptor_diffs = image_a_pred_batch - matches_b_descriptor
    norm_diffs = descriptor_diffs.norm(norm_degree, 2).pow(2)
    p_a = F.softmax(-1 * norm_diffs, dim=1).double() # compute current distribution
    q_a = gauss_2d_batch(image_width, image_height, sigma, matches_a%image_width, matches_a/image_width, masked_indices=masked_indices)
    if symm_matches_a is not None:
	q_a_symm = gauss_2d_batch(image_width, image_height, sigma, symm_matches_a%image_width, symm_matches_a/image_width, masked_indices=masked_indices)
	q_a = bimodal_gauss(q_a, q_a_symm)
    q_a = normalize(q_a)
    q_a += 1e-300
    loss = F.kl_div(q_a.log(), p_a, None, None, 'sum')/matches_b_descriptor.shape[0]
    return loss

def local_crop(U, V): 
    '''Takes in U: (201,), V: (201,), outputs (201,8), (201,8)'''
    U = U.repeat(8, 1).view(U.shape[0], 8)
    V = V.repeat(8, 1).view(V.shape[0], 8)
    x = torch.Tensor([0, 0, -1, -1, -1, 1, 1, 1]).cuda()
    y = torch.Tensor([-1, 1, 0, 1, -1, 0, 1, -1]).cuda()
    res_x = U[:,].float() + x
    res_y = V[:,].float() + y
    return res_x, res_y

def knn(k, data, points):
    data = data.repeat(points.shape[0], 1).view(points.shape[0], data.shape[0], data.shape[1])
    data = data.cuda().float()
    points = points.view(points.shape[0], 1, points.shape[1])
    dist = torch.norm(data - points.float(), dim=2, p=None)
    knn = dist.topk(k, largest=False)
    result = data[0, knn.indices]
    res_x, res_y = result[:, :, 0], result[:, :, 1]
    return res_x, res_y

def lipschitz_batch(matches_b, image_a_pred, image_b_pred, L, k, mu, image_width=640, image_height=480):
    '''
    N = length of matches_b (num annotations)
    L, mu = Lipschitz params
    k = used in kNN (to get local crop) around each match in matches_b
    '''
    # Compute descriptors for each match in matches_b
    matches_b_descriptor = torch.index_select(image_b_pred, 1, matches_b) 
    matches_b_descriptor = matches_b_descriptor.view(matches_b_descriptor.shape[1], 1, matches_b_descriptor.shape[2])
    norm_degree = 2
    image_a_pred_batch = image_a_pred.squeeze().repeat(matches_b.shape[0], 1).view(matches_b.shape[0], image_a_pred.shape[1], image_a_pred.shape[2])
    descriptor_diffs = image_a_pred_batch - matches_b_descriptor
    norm_diffs = descriptor_diffs.norm(norm_degree, 2).pow(2)
    # Get the raw best matches for matches_b in image A
    pred_match_a_indices = torch.argmin(norm_diffs, dim=1) # in image A!!!
    pred_matches_a_U = pred_match_a_indices%image_width
    pred_matches_a_V = pred_match_a_indices/image_width
    pred_matches_a = torch.cat((pred_matches_a_U, pred_matches_a_V)).repeat(1,k).view(-1,2)

    matches_b_U = matches_b%image_width
    matches_b_V = matches_b/image_width
    # Tile matches_b so that each pixel is repeated K times (used later on in computing ground truth distance between matches_b and its neighbors)
    matches_b = torch.cat((matches_b_U, matches_b_V)).repeat(1,k).view(-1,2)

    # Get the local crop PER match_b in matches_b 
    neighbors_b_U, neighbors_b_V = knn(k, torch.ones((image_width,image_height)).nonzero().float(), torch.cat((matches_b_U, matches_b_V)).view(-1,2))

    neighbors_b = torch.cat((neighbors_b_U, neighbors_b_V)).view(-1, 2)
    neighbors_b_indices = neighbors_b_U%image_width + neighbors_b_V*image_width
    neighbors_b_indices = torch.clamp(neighbors_b_indices.long().flatten(), 0, image_width*image_height - 1)
    # Get descriptors for all neighboring pixels (used for calculating each of their best matches in image A)
    neighbors_b_descriptor = torch.index_select(image_b_pred, 1, neighbors_b_indices)
    neighbors_b_descriptor = neighbors_b_descriptor.view(neighbors_b_descriptor.shape[1], 1, 3)

    image_a_pred_batch = image_a_pred.squeeze().repeat(neighbors_b_descriptor.shape[0], 1).view(neighbors_b_descriptor.shape[0], image_a_pred.shape[1], image_a_pred.shape[2])
    # For all N*K neighbors, we compute the descriptor difference in image A
    neighbor_descriptor_diffs = image_a_pred_batch - neighbors_b_descriptor
    neighbor_norm_diffs = neighbor_descriptor_diffs.norm(norm_degree, 2).pow(2)

    # Get the best match predictions for neighbors_b
    pred_match_a_neighbor_idxs = torch.argmin(neighbor_norm_diffs, dim=1)
    pred_neighbors_a_U = pred_match_a_neighbor_idxs%image_width 
    pred_neighbors_a_V = pred_match_a_neighbor_idxs/image_width 
    pred_neighbors_a = torch.cat((pred_neighbors_a_U, pred_neighbors_a_V)).view(-1, 2)
   
    # Enforce that ||pred_matches_a - pred_neighbors_a|| <= L(||matches_b - neighbors_b||)
    # --> final loss = relu(mu(||pred_matches_a - pred_neighbors_a|| - L||matches_b - neighbors_b||))
    L_a = torch.sqrt((pred_matches_a - pred_neighbors_a).pow(2).sum(1).double())
    L_b = torch.sqrt((matches_b.float() - neighbors_b).pow(2).sum(1).double())
    loss = max(mu*(L_a - L * L_b).sum(), 0)
    return loss
    
def get_distributional_loss(image_a_pred, image_b_pred, image_a_mask, image_b_mask,  matches_a, matches_b, bimodal=False):
    L_lip_a_b = lipschitz_batch(matches_b, image_a_pred, image_b_pred, 1, 10, 0.005)
    L_lip_b_a = lipschitz_batch(matches_a, image_b_pred, image_a_pred, 1, 10, 0.005)
    masked_indices_a = flattened_mask_indices(image_a_mask, inverse=True)
    masked_indices_b = flattened_mask_indices(image_b_mask, inverse=True)
    reverse_idxs = list(range(len(matches_a)-1, -1, -1))
    #symm_matches_a = matches_a.index_select(0, torch.LongTensor(reverse_idxs).cuda()) 
    #symm_matches_b = matches_b.index_select(0, torch.LongTensor(reverse_idxs).cuda()) 
    symm_matches_a = None
    symm_matches_b = None
    L_a_b = distributional_loss_batch(image_a_pred, image_b_pred, matches_a, matches_b, masked_indices=masked_indices_a, symm_matches_a=symm_matches_a)
    L_b_a = distributional_loss_batch(image_b_pred, image_a_pred, matches_b, matches_a, masked_indices=masked_indices_b, symm_matches_a=symm_matches_b)
    lipschitz = 0.5*L_lip_a_b + 0.5*L_lip_b_a
    distributional = 0.5*L_a_b + 0.5*L_b_a 
    total_loss = lipschitz + distributional
    return total_loss, distributional, lipschitz

def get_within_scene_loss(pixelwise_contrastive_loss, image_a_pred, image_b_pred,
                                        matches_a,    matches_b,
                                        masked_non_matches_a, masked_non_matches_b,
                                        background_non_matches_a, background_non_matches_b,
                                        blind_non_matches_a, blind_non_matches_b):
    """
    Simple wrapper for pixelwise_contrastive_loss functions.  Args and return args documented above in get_loss()
    """
    pcl = pixelwise_contrastive_loss

    match_loss, masked_non_match_loss, num_masked_hard_negatives =\
        pixelwise_contrastive_loss.get_loss_matched_and_non_matched_with_l2(image_a_pred,         image_b_pred,
                                                                          matches_a,            matches_b,
                                                                          masked_non_matches_a, masked_non_matches_b,
                                                                          M_descriptor=pcl._config["M_masked"])

    if pcl._config["use_l2_pixel_loss_on_background_non_matches"]:
        background_non_match_loss, num_background_hard_negatives =\
            pixelwise_contrastive_loss.non_match_loss_with_l2_pixel_norm(image_a_pred, image_b_pred, matches_b, 
                background_non_matches_a, background_non_matches_b, M_descriptor=pcl._config["M_background"])    
        
    else:
        background_non_match_loss, num_background_hard_negatives =\
            pixelwise_contrastive_loss.non_match_loss_descriptor_only(image_a_pred, image_b_pred,
                                                                    background_non_matches_a, background_non_matches_b,
                                                                    M_descriptor=pcl._config["M_background"])
        
        

    blind_non_match_loss = zero_loss()
    num_blind_hard_negatives = 1
    if not (SpartanDataset.is_empty(blind_non_matches_a.data)):
        blind_non_match_loss, num_blind_hard_negatives =\
            pixelwise_contrastive_loss.non_match_loss_descriptor_only(image_a_pred, image_b_pred,
                                                                    blind_non_matches_a, blind_non_matches_b,
                                                                    M_descriptor=pcl._config["M_masked"])
        


    total_num_hard_negatives = num_masked_hard_negatives + num_background_hard_negatives
    total_num_hard_negatives = max(total_num_hard_negatives, 1)

    if pcl._config["scale_by_hard_negatives"]:
        scale_factor = total_num_hard_negatives

        masked_non_match_loss_scaled = masked_non_match_loss*1.0/max(num_masked_hard_negatives, 1)

        background_non_match_loss_scaled = background_non_match_loss*1.0/max(num_background_hard_negatives, 1)

        blind_non_match_loss_scaled = blind_non_match_loss*1.0/max(num_blind_hard_negatives, 1)
    else:
        # we are not currently using blind non-matches
        num_masked_non_matches = max(len(masked_non_matches_a),1)
        num_background_non_matches = max(len(background_non_matches_a),1)
        num_blind_non_matches = max(len(blind_non_matches_a),1)
        scale_factor = num_masked_non_matches + num_background_non_matches


        masked_non_match_loss_scaled = masked_non_match_loss*1.0/num_masked_non_matches

        background_non_match_loss_scaled = background_non_match_loss*1.0/num_background_non_matches

        blind_non_match_loss_scaled = blind_non_match_loss*1.0/num_blind_non_matches



    non_match_loss = 1.0/scale_factor * (masked_non_match_loss + background_non_match_loss)

    loss = pcl._config["match_loss_weight"] * match_loss + \
    pcl._config["non_match_loss_weight"] * non_match_loss

    

    return loss, match_loss, masked_non_match_loss_scaled, background_non_match_loss_scaled, blind_non_match_loss_scaled

def get_within_scene_loss_triplet(pixelwise_contrastive_loss, image_a_pred, image_b_pred,
                                        matches_a,    matches_b,
                                        masked_non_matches_a, masked_non_matches_b,
                                        background_non_matches_a, background_non_matches_b,
                                        blind_non_matches_a, blind_non_matches_b):
    """
    Simple wrapper for pixelwise_contrastive_loss functions.  Args and return args documented above in get_loss()
    """
    
    pcl = pixelwise_contrastive_loss

    masked_triplet_loss =\
        pixelwise_contrastive_loss.get_triplet_loss(image_a_pred, image_b_pred, matches_a, 
            matches_b, masked_non_matches_a, masked_non_matches_b, pcl._config["alpha_triplet"])
        
    background_triplet_loss =\
        pixelwise_contrastive_loss.get_triplet_loss(image_a_pred, image_b_pred, matches_a, 
            matches_b, background_non_matches_a, background_non_matches_b, pcl._config["alpha_triplet"])

    total_loss = masked_triplet_loss + background_triplet_loss

    return total_loss, zero_loss(), zero_loss(), zero_loss(), zero_loss()

def get_different_object_loss(pixelwise_contrastive_loss, image_a_pred, image_b_pred,
                              blind_non_matches_a, blind_non_matches_b):
    """
    Simple wrapper for pixelwise_contrastive_loss functions.  Args and return args documented above in get_loss()
    """

    scale_by_hard_negatives = pixelwise_contrastive_loss.config["scale_by_hard_negatives_DIFFERENT_OBJECT"]
    blind_non_match_loss = zero_loss()
    if not (SpartanDataset.is_empty(blind_non_matches_a.data)):
        M_descriptor = pixelwise_contrastive_loss.config["M_background"]

        blind_non_match_loss, num_hard_negatives =\
            pixelwise_contrastive_loss.non_match_loss_descriptor_only(image_a_pred, image_b_pred,
                                                                    blind_non_matches_a, blind_non_matches_b,
                                                                    M_descriptor=M_descriptor)
        
        if scale_by_hard_negatives:
            scale_factor = max(num_hard_negatives, 1)
        else:
            scale_factor = max(len(blind_non_matches_a), 1)

        blind_non_match_loss = 1.0/scale_factor * blind_non_match_loss
    loss = blind_non_match_loss
    return loss, zero_loss(), zero_loss(), zero_loss(), blind_non_match_loss

def get_same_object_across_scene_loss(pixelwise_contrastive_loss, image_a_pred, image_b_pred,
                              blind_non_matches_a, blind_non_matches_b):
    """
    Simple wrapper for pixelwise_contrastive_loss functions.  Args and return args documented above in get_loss()
    """
    blind_non_match_loss = zero_loss()
    if not (SpartanDataset.is_empty(blind_non_matches_a.data)):
        blind_non_match_loss, num_hard_negatives =\
            pixelwise_contrastive_loss.non_match_loss_descriptor_only(image_a_pred, image_b_pred,
                                                                    blind_non_matches_a, blind_non_matches_b,
                                                                    M_descriptor=pcl._config["M_masked"], invert=True)

    if pixelwise_contrastive_loss._config["scale_by_hard_negatives"]:
        scale_factor = max(num_hard_negatives, 1)
    else:
        scale_factor = max(len(blind_non_matches_a), 1)

    loss = 1.0/scale_factor * blind_non_match_loss
    blind_non_match_loss_scaled = 1.0/scale_factor * blind_non_match_loss
    return loss, zero_loss(), zero_loss(), zero_loss(), blind_non_match_loss

def zero_loss():
    return Variable(torch.FloatTensor([0]).cuda())

def is_zero_loss(loss):
    return loss.data[0] < 1e-20


