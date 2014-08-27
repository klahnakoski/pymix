import mixtureLinearGaussian
import unittest
import mixture
import numpy
import random
import copy

class LinearGaussianDistributionTests(unittest.TestCase):
	"""
	Tests for class LinearGaussianDistribution
	"""

	def setUp(self):
		self.dist = mixtureLinearGaussian.LinearGaussianDistribution(2,[0.6,0.7],[1.0])

	def testeq(self):
		tdist1 = mixtureLinearGaussian.LinearGaussianDistribution(2,[0.6,0.7],[1.0])
		self.assertEqual(self.dist,tdist1)

		tdist2 = mixtureLinearGaussian.LinearGaussianDistribution(2,[0.55,0.33],[0.7])
		self.assertNotEqual(self.dist,tdist2)

	def testcopy(self):
		cp = copy.copy(self.dist)

		self.dist.beta = [0.5,0.87]
		self.assertEqual(str(self.dist.beta),'[0.5, 0.87]')
		self.dist.sigma = [3.2]
		self.assertEqual(self.dist.sigma,[3.2])

		self.assertEqual(str(cp.beta),'[ 0.6  0.7]')
		self.assertEqual(cp.sigma,[1.0])

	def testpdf(self):
		a = numpy.array([[0.5,1.0],[7.2,3.4],[12.2,9.4],[4.6,3.7],[5.5,1.5],[4.2,2.7]], dtype='Float64')
		p =  self.dist.pdf(a)                
		self.assertEqual(str(p),'[ -0.27893853  16.88946147  24.28146147   1.06916147  13.90356147\n   2.00516147]')

	def testmstep(self):
		a = numpy.array([[0.5,1.0],[7.2,3.4],[12.2,9.4],[4.6,3.7],[5.5,1.5],[4.2,2.7]], dtype='Float64')
		post1 = numpy.array([ 0.5,0.5,0.5,0.5,0.5,0.5 ])
		self.dist.MStep(post1,a)
		self.assertEqual(str(self.dist.beta),'[ 1.53596949  1.15134484]')
		self.assertEqual(str(self.dist.sigma[0]),'1.55468583309')

	def testsample(self):
		random.seed(3586662)
		x = self.dist.sample()
		self.assertEqual(str(x),'[-1.8603178432821363, -1.1305815609985377]')

	def testsampleset(self):
		random.seed(3586662)
		x = self.dist.sampleSet(5)
		self.assertEqual(str(x),'[[-1.86031784 -1.13058156]\n [-2.17957541 -3.05726151]\n [-0.96864211 -1.57436875]\n [-0.75192407  0.8928686 ]\n [ 1.20995366  0.17397463]]')	
	
if __name__ == '__main__':
    unittest.main()
