import torch

def knn(k, data, points):
	''' 
	Gets the k nearest neighbors from given data for a set of points
	data.shape = (width*height, 2) # List of all pixels in a given image
	points.shape = (N, 2) # Set of N pixels in image
	result.shape = (N, K, 2) # K nearest neighbors for each of N input points
	'''
	data = data.repeat(points.shape[0], 1).view(points.shape[0], data.shape[0], data.shape[1])
	points = points.view(points.shape[0], 1, points.shape[1])
	dist = torch.norm(data - points, dim=2, p=None)
	knn = dist.topk(k, largest=False)
	result = data[0, knn.indices]
	print('points')
	print(points)
	print('top k=%d matches: ' % k)
	print(result)
	return result

if __name__ == '__main__':
    width = 640
    height = 480
    pixels = torch.ones((width,height)).nonzero().float()
    points = torch.Tensor([[321, 240], [200, 500], [300, 12]])
    k = 20
    knn(20, pixels, points)
