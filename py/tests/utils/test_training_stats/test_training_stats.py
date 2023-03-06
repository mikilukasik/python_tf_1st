import unittest
from unittest.mock import patch
import numpy as np

from utils.helpers.training_stats import TrainingStats


class TestTrainingStats(unittest.TestCase):

    def setUp(self):
        self.stats = TrainingStats(0.001, 32)

    def test_initial_values(self):
        self.assertEqual(self.stats.lr, 0.001)
        self.assertEqual(self.stats.batch_size, 32)
        self.assertEqual(self.stats.best_loss, np.inf)
        self.assertEqual(self.stats.lr_history, [])
        self.assertEqual(self.stats.loss_history, [])
        self.assertEqual(self.stats.time_history, [])
        self.assertEqual(self.stats.sample_size_history, [])
        self.assertEqual(self.stats.batch_size_history, [])

    def test_add_to_stats(self):
        self.stats.add_to_stats(0.1, 0.001, 10.0, 50000, 32)
        self.assertEqual(self.stats.lr, 0.001)
        self.assertEqual(self.stats.batch_size, 32)
        self.assertEqual(self.stats.best_loss, 0.1)
        self.assertEqual(self.stats.lr_history, [0.001])
        self.assertEqual(self.stats.loss_history, [0.1])
        self.assertEqual(self.stats.time_history, [10.0])
        self.assertEqual(self.stats.sample_size_history, [50000])
        self.assertEqual(self.stats.batch_size_history, [32])

    @patch('numpy.diff')
    def test_get_next_lr(self, mock_diff):
        mock_diff.return_value = [-0.01, -0.02, 0.03]
        self.stats.loss_history = [0.1, 0.09, 0.07, 0.1, 0.12, 0.15]
        self.stats.lr_history = [0.001, 0.0005,
                                 0.0001, 0.00005, 0.000025, 0.0000125]
        lr, log = self.stats.get_next_lr()
        self.assertAlmostEqual(lr, 0.000025 * 2 ** 3, places=6)
        self.assertEqual(
            log, 'Changing the learning rate from 0.000025 to 0.000200 (ideal_lr), expected improvement: 33.333333%')

    @patch('numpy.diff')
    def test_get_next_lr_best_lr(self, mock_diff):
        mock_diff.return_value = [-0.01, -0.02, 0.03]
        self.stats.loss_history = [0.1, 0.09, 0.07, 0.1, 0.12, 0.15]
        self.stats.lr_history = [0.001, 0.0005,
                                 0.0001, 0.00005, 0.000025, 0.0000125]
        self.stats.best_loss = 0.08
        lr, log = self.stats.get_next_lr()
        self.assertAlmostEqual(lr, np.sqrt(0.000025 * 0.08), places=6)
        self.assertEqual(
            log, 'Changing the learning rate from 0.000025 to 0.000041 (ideal_lr_sgd), expected improvement: 33.333333%')

    def test_get_next_lr_initial(self):
        self.stats.lr_history = [0.001]
        self.stats.loss_history = [0.1]
        lr, log = self.stats.get_next_lr()
        self.assertAlmostEqual(lr, 0.001, places=6)
        self.assertEqual(log, 'Using the initial learning rate: 0.001000')

    def test_add_to_stats():
        stats = TrainingStats(0.01, 32)
        stats.add_to_stats(0.5, 0.01, 10, 5000, 32)

        assert stats.loss_history == [0.5]
        assert stats.lr_history == [0.01]
        assert stats.time_history == [10]
        assert stats.sample_size_history == [5000]
        assert stats.batch_size_history == [32]
        assert stats.lr == 0.01
        assert stats.batch_size == 32

        stats.add_to_stats(0.3, 0.005, 20, 10000, 64)

        assert stats.loss_history == [0.5, 0.3]
        assert stats.lr_history == [0.01, 0.005]
        assert stats.time_history == [10, 20]
        assert stats.sample_size_history == [5000, 10000]
        assert stats.batch_size_history == [32, 64]
        assert stats.lr == 0.005
        assert stats.batch_size == 64

    def test_get_next_lr(self):
        stats = TrainingStats(initial_lr=0.1, initial_batch_size=16)

        # Test when history length is less than 2
        next_lr, log = stats.get_next_lr()
        self.assertAlmostEqual(next_lr, 0.1, places=5)
        self.assertEqual(log, 'Using the initial learning rate: 0.100000')

        # Test when ideal_lr < lr and ideal_lr_sgd < lr
        stats.add_to_stats(loss=0.3, lr=0.1, time=10.0,
                           sample_size=16, batch_size=16)
        stats.add_to_stats(loss=0.2, lr=0.05, time=20.0,
                           sample_size=16, batch_size=16)
        next_lr, log = stats.get_next_lr()
        self.assertAlmostEqual(next_lr, 0.025, places=5)
        self.assertIn('Changing the learning rate', log)
        self.assertIn('ideal_lr_sgd', log)

        # Test when ideal_lr < lr and ideal_lr_sgd > lr
        stats.add_to_stats(loss=0.1, lr=0.025, time=30.0,
                           sample_size=16, batch_size=16)
        next_lr, log = stats.get_next_lr()
        self.assertAlmostEqual(next_lr, 0.025, places=5)
        self.assertIn('Changing the learning rate', log)
        self.assertIn('ideal_lr', log)

        # Test when ideal_lr > lr and ideal_lr_sgd < lr
        stats.add_to_stats(loss=0.05, lr=0.025, time=40.0,
                           sample_size=16, batch_size=16)
        next_lr, log = stats.get_next_lr()
        self.assertAlmostEqual(next_lr, 0.05, places=5)
        self.assertIn('Changing the learning rate', log)
        self.assertIn('previous', log)

    def test_get_history(self):
        stats = TrainingStats(0.01, 32)
        stats.add_to_stats(1.0, 0.01, 100.0, 50000, 32)
        stats.add_to_stats(0.8, 0.01, 200.0, 50000, 32)
        stats.add_to_stats(0.6, 0.005, 300.0, 50000, 64)

        expected_history = {
            'lr_history': [0.01, 0.01, 0.005],
            'loss_history': [1.0, 0.8, 0.6],
            'time_history': [100.0, 200.0, 300.0],
            'sample_size_history': [50000, 50000, 50000],
            'batch_size_history': [32, 32, 64],
            'best_loss': 0.6,
            'lr': 0.005,
            'batch_size': 64
        }

        assert stats.get_history() == expected_history

    def test_add_to_stats(self):
        stats = TrainingStats(0.01, 64)
        stats.add_to_stats(1.0, 0.01, 10, 100, 64)
        history = stats.get_history()
        self.assertEqual(history['loss_history'], [1.0])
        self.assertEqual(history['lr_history'], [0.01])
        self.assertEqual(history['time_history'], [10])
        self.assertEqual(history['sample_size_history'], [100])
        self.assertEqual(history['batch_size_history'], [64])
        self.assertEqual(history['best_loss'], 1.0)
        self.assertEqual(history['lr'], 0.01)
        self.assertEqual(history['batch_size'], 64)

    def test_get_next_lr(self):
        stats = TrainingStats(0.01, 64)
        next_lr, log = stats.get_next_lr()
        self.assertEqual(next_lr, 0.01)
        self.assertEqual(log, 'Using the initial learning rate: 0.010000')

        stats.add_to_stats(1.0, 0.01, 10, 100, 64)
        next_lr, log = stats.get_next_lr()
        self.assertEqual(next_lr, 0.01)
        self.assertEqual(log, 'Using the initial learning rate: 0.010000')

        stats.add_to_stats(0.5, 0.01, 20, 200, 64)
        next_lr, log = stats.get_next_lr()
        self.assertAlmostEqual(next_lr, 0.02, places=5)
        self.assertEqual(
            log[:23], 'Changing the learning rate from 0.010000 to 0.020000')

        stats.add_to_stats(0.2, 0.02, 30, 300, 64)
        next_lr, log = stats.get_next_lr()
        self.assertAlmostEqual(next_lr, 0.04, places=5)
        self.assertEqual(
            log[:23], 'Changing the learning rate from 0.020000 to 0.040000')

        stats.add_to_stats(0.1, 0.04, 40, 400, 64)
        next_lr, log = stats.get_next_lr()
        self.assertAlmostEqual(next_lr, 0.08, places=5)
        self.assertEqual(
            log[:23], 'Changing the learning rate from 0.040000 to 0.080000')

        stats.add_to_stats(0.05, 0.08, 50, 500, 64)
        next_lr, log = stats.get_next_lr()
        self.assertAlmostEqual(next_lr, 0.08, places=5)
        self.assertEqual(
            log[:29], 'Changing the learning rate from 0.080000 to 0.080000 (ideal_lr_sgd)')

        stats.add_to_stats(0.01, 0.08, 60, 600, 64)
        next_lr, log = stats.get_next_lr()
        self.assertAlmostEqual(next_lr, 0.08, places=5)
        self.assertEqual(log[:22], 'Using the initial learning rate: 0.080000')
