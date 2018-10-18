import torchvision.transforms as transform
import random


# Apply at a random probability a random number of randomly chosen transformations
class RandomlyApply(transform.RandomApply):
    def __init__(self, transforms, p=0.5):
        super(RandomlyApply, self).__init__(transforms)
        self.p = p
        self.toloop = random.randint(0, len(self.transforms) - 1)
        self.loopthru = random.sample(range(0, self.toloop + 1), self.toloop)

    def __call__(self, img):
        if self.p < random.random():
            return img
        else:
            for i in range(0, self.toloop):
                img = self.transforms[self.loopthru[i]](img)

        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n    p={}'.format(self.p)
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
