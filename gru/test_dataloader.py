
import unittest

from dataloader import GowallaLoader, Split, Usage
 
class TestGowallaLoader(unittest.TestCase):
    
    def test_poi_dataset(self):
        loader = GowallaLoader(10, 201)
        loader.load('../../dataset/small-10000.txt')
        dataset = loader.poi_dataset(5, 10, Split.TEST, Usage.MAX_SEQ_LENGTH)
        self.assertEqual(84, len(dataset))
        x, times, coords, y = dataset.__getitem__(0)
        self.assertEqual(5, len(x))
 
    def test_load(self):
        loader = GowallaLoader(10, 201)
        #loader.load('../../dataset/loc-gowalla_totalCheckins.txt')
        loader.load('../../dataset/small-10000.txt')
        print('selected users: ', loader.user2id, loader.users)
        self.assertEqual(10, len(loader.users))
        self.assertEqual(10, len(loader.locs))
        self.assertEqual(225, len(loader.locs[0]))

        
if __name__ == '__main__':
    unittest.main()