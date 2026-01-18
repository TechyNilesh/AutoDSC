from river import metrics

__all__ = ["Purity"]

class Purity(metrics.base.MultiClassMetric):

	@property
	def works_with_weights(self):
		return False

	def get(self):
		purity = 0
		num = 0
		#print(self.cm)
		for j in self.cm.classes:  # predict
			maxi = 0
			for i in self.cm.classes:  # label
				num += self.cm[i][j]
				if maxi < self.cm[i][j]:
					maxi = self.cm[i][j]
			purity += maxi

		return purity / num