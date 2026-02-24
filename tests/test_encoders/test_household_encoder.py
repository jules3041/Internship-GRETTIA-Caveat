"""
Unit tests for HouseholdEncoder
Tests encoding and decoding of household-level activity schedules
"""
import pytest
import pandas as pd
import torch
import numpy as np
from pathlib import Path

from caveat.encoding.household_encoder import HouseholdEncoder
from caveat.data.household_dataset import HouseholdDataset


@pytest.fixture
def household_schedules():
    """Load test household schedules with multiple members per household"""
    data_path = Path("tests/fixtures/household_test_schedules.csv")
    schedules = pd.read_csv(data_path)
    return schedules


@pytest.fixture
def household_attributes():
    """Load test household attributes"""
    data_path = Path("tests/fixtures/household_test_attributes.csv")
    attributes = pd.read_csv(data_path)
    return attributes


@pytest.fixture
def household_config(tmp_path):
    """Configuration for household encoder with test data"""
    return {
        "time_steps": 144,  # 10-minute intervals
        "attributes_path": "tests/fixtures/household_test_attributes.csv",
        "dataset_params": {
            "max_members": 5,
            "group_by": "hid",
            "label_cols": ["age", "sex", "employed", "household_size"]
        }
    }


class TestHouseholdEncoder:
    """Test suite for HouseholdEncoder"""

    def test_encoder_initialization(self, household_config):
        """Test that encoder initializes correctly"""
        encoder = HouseholdEncoder(**household_config)
        assert encoder.time_steps == 144
        assert encoder.config == household_config
        assert isinstance(encoder.idx_to_act, dict)

    def test_encode_creates_dataset(self, household_schedules, household_attributes, household_config):
        """Test that encode method creates a HouseholdDataset"""
        encoder = HouseholdEncoder(**household_config)
        dataset = encoder.encode(household_schedules)

        # Check dataset type
        assert isinstance(dataset, HouseholdDataset)

        # Check dataset has the right number of households
        # h1: 2 members, h2: 3 members, h3: 1 member = 3 households
        assert len(dataset) == 3

        # Check activity mapping was created
        assert hasattr(encoder, 'idx_to_act')
        assert len(encoder.idx_to_act) > 0
        assert 'PAD' in dataset.act_to_idx
        assert dataset.act_to_idx['PAD'] == 0

    def test_dataset_activity_encoding(self, household_schedules, household_config):
        """Test that activities are encoded correctly"""
        encoder = HouseholdEncoder(**household_config)
        dataset = encoder.encode(household_schedules)

        # Check that all activities from schedules are encoded
        unique_acts = household_schedules['act'].unique()
        for act in unique_acts:
            assert act in dataset.act_to_idx
            assert dataset.act_to_idx[act] > 0  # PAD is 0, real activities start at 1

        # Check reverse mapping exists
        assert dataset.idx_to_act[0] == 'PAD'
        for act, idx in dataset.act_to_idx.items():
            if act != 'PAD':
                assert dataset.idx_to_act[idx] == act

    def test_dataset_label_encoding(self, household_schedules, household_attributes, household_config):
        """Test that labels (attributes) are encoded correctly"""
        encoder = HouseholdEncoder(**household_config)
        dataset = encoder.encode(household_schedules)

        # Check label dictionaries were created
        label_cols = household_config['dataset_params']['label_cols']
        assert len(dataset.label_dics) == len(label_cols)

        for col in label_cols:
            assert col in dataset.label_dics
            assert 'PAD' in dataset.label_dics[col]
            assert dataset.label_dics[col]['PAD'] == 0

        # Check cardinalities
        assert len(dataset.label_cardinalities) == len(label_cols)
        assert all(card > 1 for card in dataset.label_cardinalities)  # At least PAD + 1 value

    def test_dataset_batch_shape(self, household_schedules, household_config):
        """Test that batched data has correct shape"""
        encoder = HouseholdEncoder(**household_config)
        dataset = encoder.encode(household_schedules)

        # Get first household
        batch = dataset[0]

        # Check batch structure: dict with "x", "labels", "mask"
        assert isinstance(batch, dict)
        assert "x" in batch
        assert "labels" in batch
        assert "mask" in batch

        # Check shapes
        x = batch["x"]
        labels = batch["labels"]
        mask = batch["mask"]

        max_members = household_config['dataset_params']['max_members']
        time_steps = household_config['time_steps']
        n_label_cols = len(household_config['dataset_params']['label_cols'])

        assert x.shape == (max_members, time_steps)
        assert labels.shape == (max_members, n_label_cols)
        assert mask.shape == (max_members,)

        # Check data types
        assert x.dtype == torch.long
        assert labels.dtype == torch.long
        assert mask.dtype == torch.float

    def test_household_padding(self, household_schedules, household_config):
        """Test that households with fewer members are padded correctly"""
        encoder = HouseholdEncoder(**household_config)
        dataset = encoder.encode(household_schedules)

        # h3 has only 1 member, should be padded to 5
        # Find the household with 1 member (p6 in h3)
        batch_h3 = dataset[2]  # Third household (index 2)

        mask = batch_h3["mask"]
        x = batch_h3["x"]

        # First member should be active (mask=1)
        assert mask[0] == 1.0

        # Other members should be padded (mask=0)
        assert torch.all(mask[1:] == 0.0)

        # Padded members should have all PAD activities (index 0)
        assert torch.all(x[1:, :] == 0)

    def test_household_member_ordering(self, household_schedules, household_config):
        """Test that household members are sorted by age (descending) then sex"""
        encoder = HouseholdEncoder(**household_config)
        dataset = encoder.encode(household_schedules)

        # h2 has 3 members: ages 38, 12, 8
        # After sorting by age desc: 38 (F), 12 (M), 8 (F)
        batch_h2 = dataset[1]  # Second household
        mask = batch_h2["mask"]

        # All 3 members should be active
        assert torch.sum(mask) == 3.0
        assert mask[0] == 1.0  # Oldest member
        assert mask[1] == 1.0  # Middle member
        assert mask[2] == 1.0  # Youngest member
        assert mask[3] == 0.0  # Padded
        assert mask[4] == 0.0  # Padded

    def test_decode_simple(self, household_config):
        """Test decoding of simple synthetic schedules"""
        encoder = HouseholdEncoder(**household_config)

        # Create a simple activity mapping
        encoder.idx_to_act = {0: "PAD", 1: "home", 2: "work", 3: "shop"}

        # Create synthetic schedule tensor: 2 households, 5 members each, 144 time steps
        # Shape: [2, 5, 144]
        schedules = torch.zeros(2, 5, 144, dtype=torch.long)

        # Household 1, Member 1: home (0-72), work (72-144)
        schedules[0, 0, :72] = 1  # home
        schedules[0, 0, 72:] = 2  # work

        # Household 1, Member 2: all PAD (padded member)
        schedules[0, 1, :] = 0

        # Household 2, Member 1: home all day
        schedules[1, 0, :] = 1

        # Decode
        result = encoder.decode(schedules, argmax=False)

        # Check result is a DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

        # Check columns
        expected_cols = ['pid', 'hid', 'member_idx', 'act', 'start', 'end', 'duration']
        assert all(col in result.columns for col in expected_cols)

        # Check that PAD activities are excluded
        assert 'PAD' not in result['act'].values

        # Check household and member indexing
        assert 'Gen_HH0_M0' in result['pid'].values
        assert 'Gen_HH1_M0' in result['pid'].values

        # Padded member should not appear in result
        assert 'Gen_HH0_M1' not in result['pid'].values

    def test_decode_with_argmax(self, household_config):
        """Test decoding with argmax from probability distributions"""
        encoder = HouseholdEncoder(**household_config)
        encoder.idx_to_act = {0: "PAD", 1: "home", 2: "work"}

        # Create probability tensor: [B, M, T, Classes]
        # 1 household, 2 members, 144 timesteps, 3 activity classes
        probs = torch.zeros(1, 2, 144, 3)

        # Member 1: High probability of 'home' (index 1) for first half
        probs[0, 0, :72, 1] = 0.9
        probs[0, 0, :72, 0] = 0.05
        probs[0, 0, :72, 2] = 0.05

        # Member 1: High probability of 'work' (index 2) for second half
        probs[0, 0, 72:, 2] = 0.9
        probs[0, 0, 72:, 0] = 0.05
        probs[0, 0, 72:, 1] = 0.05

        # Member 2: All PAD
        probs[0, 1, :, 0] = 1.0

        # Decode with argmax
        result = encoder.decode(probs, argmax=True)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

        # Should have 2 activities for member 0
        member_0_acts = result[result['pid'] == 'Gen_HH0_M0']
        assert len(member_0_acts) == 2
        assert 'home' in member_0_acts['act'].values
        assert 'work' in member_0_acts['act'].values

    def test_decode_preserves_time_continuity(self, household_config):
        """Test that decoded schedules have continuous time coverage"""
        encoder = HouseholdEncoder(**household_config)
        encoder.idx_to_act = {0: "PAD", 1: "home", 2: "work", 3: "leisure"}

        # Create schedule with transitions
        schedules = torch.zeros(1, 1, 144, dtype=torch.long)
        schedules[0, 0, :48] = 1   # home: 0-480 min
        schedules[0, 0, 48:120] = 2  # work: 480-1200 min
        schedules[0, 0, 120:] = 1   # home: 1200-1440 min

        result = encoder.decode(schedules, argmax=False)

        # Sort by start time
        result = result.sort_values('start')

        # Check time continuity: end of one activity = start of next
        for i in range(len(result) - 1):
            assert result.iloc[i]['end'] == result.iloc[i + 1]['start']

        # Check full day coverage
        assert result.iloc[0]['start'] == 0
        assert result.iloc[-1]['end'] == 1440

        # Check duration calculation
        for _, row in result.iterrows():
            assert row['duration'] == row['end'] - row['start']

    def test_encode_decode_roundtrip(self, household_schedules, household_config):
        """Test that encoding then decoding preserves activity sequences"""
        encoder = HouseholdEncoder(**household_config)
        dataset = encoder.encode(household_schedules)

        # Get a batch and extract the activity grid
        batch = dataset[0]
        x = batch["x"]  # Shape: [5, 144]

        # Convert to expected decode input format [1, 5, 144]
        x_batch = x.unsqueeze(0)

        # Decode
        decoded = encoder.decode(x_batch, argmax=False)

        # Check structure
        assert isinstance(decoded, pd.DataFrame)
        assert len(decoded) > 0

        # Check that activities match original activity set
        original_acts = set(household_schedules['act'].unique())
        decoded_acts = set(decoded['act'].unique())
        assert decoded_acts.issubset(original_acts)

    def test_activity_weights_calculation(self, household_schedules, household_config):
        """Test that activity weights are calculated for imbalanced data"""
        encoder = HouseholdEncoder(**household_config)
        dataset = encoder.encode(household_schedules)

        # Check weights exist
        assert hasattr(dataset, 'activity_weights')
        assert isinstance(dataset.activity_weights, torch.Tensor)

        # PAD should have weight 0
        assert dataset.activity_weights[0] == 0.0

        # Other activities should have positive weights
        for idx in range(1, len(dataset.activity_weights)):
            assert dataset.activity_weights[idx] > 0

        # Weights should be clamped between min and max
        assert torch.all(dataset.activity_weights <= 10.0)
        assert torch.all(dataset.activity_weights >= 0.0)

    def test_empty_household_handling(self, household_config):
        """Test handling of edge case with empty schedules"""
        encoder = HouseholdEncoder(**household_config)

        # Create minimal valid schedules
        schedules = pd.DataFrame({
            'pid': ['p1'],
            'hid': ['h1'],
            'act': ['home'],
            'start': [0],
            'end': [1440],
            'duration': [1440]
        })

        dataset = encoder.encode(schedules)

        # Should have 1 household
        assert len(dataset) == 1

        # Get batch
        batch = dataset[0]
        assert batch['x'].shape[0] == household_config['dataset_params']['max_members']

    def test_tensor_device_handling(self, household_config):
        """Test that decode handles tensors on different devices correctly"""
        encoder = HouseholdEncoder(**household_config)
        encoder.idx_to_act = {0: "PAD", 1: "home", 2: "work"}

        # Create tensor on CPU
        schedules_cpu = torch.zeros(1, 2, 144, dtype=torch.long)
        schedules_cpu[0, 0, :] = 1

        # Decode from CPU tensor
        result = encoder.decode(schedules_cpu, argmax=False)
        assert isinstance(result, pd.DataFrame)

        # If CUDA available, test GPU tensor
        if torch.cuda.is_available():
            schedules_gpu = schedules_cpu.cuda()
            result_gpu = encoder.decode(schedules_gpu, argmax=False)
            assert isinstance(result_gpu, pd.DataFrame)
            # Results should be the same
            pd.testing.assert_frame_equal(result, result_gpu)


class TestHouseholdDataset:
    """Additional tests specific to HouseholdDataset class"""

    def test_shape_method(self, household_schedules, household_config):
        """Test that shape() returns correct dimensions"""
        encoder = HouseholdEncoder(**household_config)
        dataset = encoder.encode(household_schedules)

        shape = dataset.shape()
        max_members = household_config['dataset_params']['max_members']
        time_steps = household_config['time_steps']

        assert shape == (max_members, time_steps)

    def test_household_grouping(self, household_schedules, household_config):
        """Test that households are grouped correctly"""
        encoder = HouseholdEncoder(**household_config)
        dataset = encoder.encode(household_schedules)

        # Check number of households
        unique_hids = household_schedules['hid'].nunique()
        assert len(dataset.households) == unique_hids

        # Each household group should have the household ID
        for hh_group in dataset.households:
            assert 'hid' in hh_group.columns
            # All members in a household should have the same hid
            assert hh_group['hid'].nunique() == 1

    def test_len_method(self, household_schedules, household_config):
        """Test that __len__ returns correct number of households"""
        encoder = HouseholdEncoder(**household_config)
        dataset = encoder.encode(household_schedules)

        expected_len = household_schedules['hid'].nunique()
        assert len(dataset) == expected_len
