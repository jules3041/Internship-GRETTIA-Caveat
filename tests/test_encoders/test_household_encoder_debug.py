"""
Debug tests for HouseholdEncoder with detailed data printing at each stage.
These tests show the complete data transformation pipeline.
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


def test_dataset_activity_encoding(household_schedules, household_config):
    """Test that activities are encoded correctly with detailed printing"""
    print("\n" + "="*80)
    print("TEST: Activity Encoding Pipeline")
    print("="*80)

    # Stage 1: Input schedules
    print("\n[STAGE 1] INPUT SCHEDULES:")
    print("-" * 80)
    print(household_schedules.to_string())
    print(f"\nUnique activities: {sorted(household_schedules['act'].unique())}")
    print(f"Total schedule records: {len(household_schedules)}")
    print(f"Unique persons: {household_schedules['pid'].nunique()}")
    print(f"Unique households: {household_schedules['hid'].nunique()}")

    # Stage 2: Create encoder
    print("\n[STAGE 2] ENCODER INITIALIZATION:")
    print("-" * 80)
    encoder = HouseholdEncoder(**household_config)
    print(f"Time steps: {encoder.time_steps}")
    print(f"Config: {encoder.config['dataset_params']}")

    # Stage 3: Encode to dataset
    print("\n[STAGE 3] ENCODING TO DATASET:")
    print("-" * 80)
    dataset = encoder.encode(household_schedules)

    # Stage 4: Activity mapping
    print("\n[STAGE 4] ACTIVITY ENCODING MAPPINGS:")
    print("-" * 80)
    print("act_to_idx mapping:")
    for act, idx in sorted(dataset.act_to_idx.items(), key=lambda x: x[1]):
        print(f"  {act:15s} -> {idx}")

    print("\nidx_to_act mapping (reverse):")
    for idx, act in sorted(dataset.idx_to_act.items()):
        print(f"  {idx} -> {act:15s}")

    print(f"\nTotal activity encodings: {dataset.activity_encodings}")

    # Stage 5: Activity weights
    print("\n[STAGE 5] ACTIVITY WEIGHTS (for loss function):")
    print("-" * 80)
    for idx, weight in enumerate(dataset.activity_weights):
        act_name = dataset.idx_to_act.get(idx, "UNKNOWN")
        print(f"  {act_name:15s} (idx={idx}): weight={weight:.4f}")

    # Stage 6: Dataset structure
    print("\n[STAGE 6] DATASET STRUCTURE:")
    print("-" * 80)
    print(f"Number of households: {len(dataset)}")
    print(f"Dataset shape (per household): {dataset.shape()}")

    # Stage 7: First household details
    print("\n[STAGE 7] FIRST HOUSEHOLD SAMPLE:")
    print("-" * 80)
    batch = dataset[0]
    print(f"Batch keys: {batch.keys()}")
    print(f"  x (activities) shape: {batch['x'].shape}")
    print(f"  labels (attributes) shape: {batch['labels'].shape}")
    print(f"  mask shape: {batch['mask'].shape}")

    print("\nActivity grid (x) for first household:")
    print(f"  Shape: [members={batch['x'].shape[0]}, timesteps={batch['x'].shape[1]}]")
    for member_idx in range(batch['x'].shape[0]):
        if batch['mask'][member_idx] == 1.0:
            activities = batch['x'][member_idx].tolist()
            unique_acts = list(set(activities))
            act_names = [dataset.idx_to_act[a] for a in unique_acts if a in dataset.idx_to_act]
            print(f"  Member {member_idx}: active={batch['mask'][member_idx].item():.0f}, activities={act_names}")
            # Show first 20 timesteps
            first_20 = activities[:20]
            first_20_names = [dataset.idx_to_act.get(a, '?')[:4] for a in first_20]
            print(f"    First 20 timesteps: {first_20_names}")
        else:
            print(f"  Member {member_idx}: PADDED")

    print("\nLabels (attributes) for first household:")
    for member_idx in range(batch['labels'].shape[0]):
        if batch['mask'][member_idx] == 1.0:
            print(f"  Member {member_idx}: {batch['labels'][member_idx].tolist()}")
        else:
            print(f"  Member {member_idx}: PADDED")

    # Assertions
    print("\n[VALIDATION]:")
    print("-" * 80)

    # Check that all activities from schedules are encoded
    unique_acts = household_schedules['act'].unique()
    for act in unique_acts:
        assert act in dataset.act_to_idx, f"Activity '{act}' not in mapping"
        assert dataset.act_to_idx[act] > 0, f"Activity '{act}' should have index > 0"
        print(f"✓ Activity '{act}' correctly encoded")

    # Check reverse mapping exists
    assert dataset.idx_to_act[0] == 'PAD', "Index 0 should be PAD"
    print("✓ PAD correctly mapped to index 0")

    for act, idx in dataset.act_to_idx.items():
        if act != 'PAD':
            assert dataset.idx_to_act[idx] == act, f"Reverse mapping failed for {act}"
    print("✓ Reverse mapping (idx_to_act) is consistent")

    print("\n" + "="*80)
    print("TEST PASSED: Activity encoding successful!")
    print("="*80 + "\n")


def test_decode_simple(household_config):
    """Test decoding of simple synthetic schedules with detailed printing"""
    print("\n" + "="*80)
    print("TEST: Decode Simple Synthetic Schedules")
    print("="*80)

    # Stage 1: Setup
    print("\n[STAGE 1] SETUP:")
    print("-" * 80)
    encoder = HouseholdEncoder(**household_config)

    # Create a simple activity mapping
    encoder.idx_to_act = {0: "PAD", 1: "home", 2: "work", 3: "shop"}
    print("Activity mapping:")
    for idx, act in sorted(encoder.idx_to_act.items()):
        print(f"  {idx} -> {act}")

    # Stage 2: Create synthetic tensor
    print("\n[STAGE 2] CREATE SYNTHETIC SCHEDULES:")
    print("-" * 80)
    # Shape: [2 households, 5 members, 144 time steps]
    schedules = torch.zeros(2, 5, 144, dtype=torch.long)

    # Household 1, Member 1: home (0-72), work (72-144)
    schedules[0, 0, :72] = 1  # home
    schedules[0, 0, 72:] = 2  # work
    print("Household 0, Member 0:")
    print(f"  Timesteps 0-71:   activity={encoder.idx_to_act[1]} (index=1)")
    print(f"  Timesteps 72-143: activity={encoder.idx_to_act[2]} (index=2)")

    # Household 1, Member 2: all PAD (padded member)
    schedules[0, 1, :] = 0
    print("Household 0, Member 1:")
    print(f"  All timesteps:    PADDED (index=0)")

    # Household 2, Member 1: home all day
    schedules[1, 0, :] = 1
    print("Household 1, Member 0:")
    print(f"  All timesteps:    activity={encoder.idx_to_act[1]} (index=1)")

    # Remaining members: PAD
    print("Household 0, Members 2-4: PADDED")
    print("Household 1, Members 1-4: PADDED")

    print(f"\nSchedules tensor shape: {schedules.shape}")
    print(f"Schedules dtype: {schedules.dtype}")

    # Stage 3: Show raw tensor for first member
    print("\n[STAGE 3] RAW TENSOR DATA (Household 0, Member 0):")
    print("-" * 80)
    member_0_acts = schedules[0, 0].tolist()
    print(f"First 20 values: {member_0_acts[:20]}")
    print(f"Values at timestep 70-74 (around transition): {member_0_acts[70:75]}")
    print(f"Last 5 values: {member_0_acts[-5:]}")

    # Stage 4: Decode
    print("\n[STAGE 4] DECODING TO EPISODE FORMAT:")
    print("-" * 80)
    result = encoder.decode(schedules, argmax=False)

    print(f"Result type: {type(result)}")
    print(f"Number of episodes: {len(result)}")
    print(f"\nColumns: {list(result.columns)}")

    # Stage 5: Show decoded results
    print("\n[STAGE 5] DECODED EPISODES:")
    print("-" * 80)
    print(result.to_string())

    # Stage 6: Analysis
    print("\n[STAGE 6] DETAILED ANALYSIS:")
    print("-" * 80)

    for _, row in result.iterrows():
        timestep_to_min = 1440 / 144
        print(f"\nPID: {row['pid']}")
        print(f"  Household: {row['hid']}, Member: {row['member_idx']}")
        print(f"  Activity: {row['act']}")
        print(f"  Start: {row['start']:.0f} min ({row['start']/60:.1f} hours)")
        print(f"  End:   {row['end']:.0f} min ({row['end']/60:.1f} hours)")
        print(f"  Duration: {row['duration']:.0f} min ({row['duration']/60:.1f} hours)")
        print(f"  Timesteps: {int(row['start']/timestep_to_min)}-{int(row['end']/timestep_to_min)}")

    # Assertions
    print("\n[VALIDATION]:")
    print("-" * 80)

    # Check result is a DataFrame
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    print("✓ Result is a DataFrame")

    assert len(result) > 0, "Result should have episodes"
    print(f"✓ Result has {len(result)} episodes")

    # Check columns
    expected_cols = ['pid', 'hid', 'member_idx', 'act', 'start', 'end', 'duration']
    assert all(col in result.columns for col in expected_cols), "Missing columns"
    print(f"✓ All expected columns present: {expected_cols}")

    # Check that PAD activities are excluded
    assert 'PAD' not in result['act'].values, "PAD should be excluded"
    print("✓ PAD activities excluded from output")

    # Check household and member indexing
    assert 'Gen_HH0_M0' in result['pid'].values, "Missing HH0_M0"
    print("✓ Household 0, Member 0 present")

    assert 'Gen_HH1_M0' in result['pid'].values, "Missing HH1_M0"
    print("✓ Household 1, Member 0 present")

    # Padded member should not appear in result
    assert 'Gen_HH0_M1' not in result['pid'].values, "Padded member should not appear"
    print("✓ Padded member correctly excluded")

    # Check Member 0 has 2 activities
    member_0_acts = result[result['pid'] == 'Gen_HH0_M0']
    assert len(member_0_acts) == 2, f"Member 0 should have 2 activities, got {len(member_0_acts)}"
    print(f"✓ Member 0 has 2 activities: {member_0_acts['act'].tolist()}")

    assert 'home' in member_0_acts['act'].values, "Member 0 should have 'home'"
    assert 'work' in member_0_acts['act'].values, "Member 0 should have 'work'"
    print("✓ Member 0 has correct activities: home and work")

    print("\n" + "="*80)
    print("TEST PASSED: Decoding successful!")
    print("="*80 + "\n")


def test_encode_decode_roundtrip(household_schedules, household_config):
    """Test that encoding then decoding preserves activity sequences"""
    print("\n" + "="*80)
    print("TEST: Encode-Decode Roundtrip")
    print("="*80)

    # Stage 1: Input data
    print("\n[STAGE 1] INPUT SCHEDULES:")
    print("-" * 80)
    print(f"Total records: {len(household_schedules)}")
    print(f"Households: {household_schedules['hid'].nunique()}")
    print(f"Persons: {household_schedules['pid'].nunique()}")

    # Show sample of input
    print("\nSample input (first 5 rows):")
    print(household_schedules.head().to_string())

    original_activities = set(household_schedules['act'].unique())
    print(f"\nOriginal unique activities: {sorted(original_activities)}")

    # Stage 2: Encode
    print("\n[STAGE 2] ENCODING:")
    print("-" * 80)
    encoder = HouseholdEncoder(**household_config)
    dataset = encoder.encode(household_schedules)

    print(f"Dataset created with {len(dataset)} households")
    print(f"Activity encodings: {dataset.activity_encodings}")
    print("\nActivity mapping:")
    for act, idx in sorted(dataset.act_to_idx.items(), key=lambda x: x[1]):
        print(f"  {act:15s} -> {idx}")

    # Stage 3: Get encoded batch
    print("\n[STAGE 3] ENCODED BATCH (First Household):")
    print("-" * 80)
    batch = dataset[0]
    x = batch["x"]  # Shape: [5, 144]
    mask = batch["mask"]

    print(f"Activity grid shape: {x.shape}")
    print(f"Mask: {mask.tolist()}")

    active_members = int(mask.sum().item())
    print(f"\nActive members: {active_members}")

    for member_idx in range(x.shape[0]):
        if mask[member_idx] == 1.0:
            acts = x[member_idx].tolist()
            unique_acts = list(set(acts))
            act_names = [dataset.idx_to_act[a] for a in unique_acts if a in dataset.idx_to_act]
            print(f"Member {member_idx}: activities = {sorted(act_names)}")

            # Show transitions
            transitions = []
            prev = acts[0]
            for t, act in enumerate(acts[1:], 1):
                if act != prev:
                    transitions.append((t, dataset.idx_to_act.get(prev, '?'), dataset.idx_to_act.get(act, '?')))
                    prev = act
            print(f"  Transitions: {transitions[:5]}")  # Show first 5 transitions

    # Stage 4: Prepare for decode
    print("\n[STAGE 4] PREPARING FOR DECODE:")
    print("-" * 80)
    # Convert to expected decode input format [1, 5, 144]
    x_batch = x.unsqueeze(0)
    print(f"Batch shape for decode: {x_batch.shape}")
    print(f"Batch dtype: {x_batch.dtype}")
    print(f"Batch min/max values: {x_batch.min().item()}/{x_batch.max().item()}")

    # Stage 5: Decode
    print("\n[STAGE 5] DECODING:")
    print("-" * 80)
    decoded = encoder.decode(x_batch, argmax=False)

    print(f"Decoded type: {type(decoded)}")
    print(f"Number of episodes: {len(decoded)}")

    # Stage 6: Show decoded output
    print("\n[STAGE 6] DECODED EPISODES:")
    print("-" * 80)
    print(decoded.to_string())

    # Stage 7: Analysis
    print("\n[STAGE 7] ROUNDTRIP ANALYSIS:")
    print("-" * 80)

    decoded_activities = set(decoded['act'].unique())
    print(f"Original activities: {sorted(original_activities)}")
    print(f"Decoded activities:  {sorted(decoded_activities)}")

    # Check activity preservation
    print("\nActivity preservation check:")
    for act in decoded_activities:
        if act in original_activities:
            print(f"  ✓ {act:15s} - present in original")
        else:
            print(f"  ✗ {act:15s} - NOT in original!")

    # Check for missing activities
    missing = original_activities - decoded_activities
    if missing:
        print(f"\nActivities in original but not decoded: {sorted(missing)}")
        print("  (This is OK if they weren't in the first household)")
    else:
        print("\n✓ All original activities present in decoded output")

    # Stage 8: Check person-level details
    print("\n[STAGE 8] PERSON-LEVEL VALIDATION:")
    print("-" * 80)

    # Get original household 1 (h1) data
    h1_schedules = household_schedules[household_schedules['hid'] == 'h1']
    print(f"Original household h1 has {h1_schedules['pid'].nunique()} members")

    for pid in h1_schedules['pid'].unique():
        original_person = h1_schedules[h1_schedules['pid'] == pid]
        original_acts = sorted(original_person['act'].unique())
        print(f"\nOriginal {pid}: activities = {original_acts}")

    print(f"\nDecoded household has {decoded['member_idx'].nunique()} members")
    for member_idx in decoded['member_idx'].unique():
        decoded_person = decoded[decoded['member_idx'] == member_idx]
        decoded_acts = sorted(decoded_person['act'].unique())
        print(f"Decoded member {member_idx}: activities = {decoded_acts}")

    # Assertions
    print("\n[VALIDATION]:")
    print("-" * 80)

    # Check structure
    assert isinstance(decoded, pd.DataFrame), "Decoded should be DataFrame"
    print("✓ Decoded result is a DataFrame")

    assert len(decoded) > 0, "Decoded should have episodes"
    print(f"✓ Decoded has {len(decoded)} episodes")

    # Check that activities match original activity set
    assert decoded_activities.issubset(original_activities), "Decoded activities should be subset of original"
    print("✓ All decoded activities are from the original activity set")

    # Check time continuity for each person
    print("\nTime continuity check:")
    for member_idx in decoded['member_idx'].unique():
        member_data = decoded[decoded['member_idx'] == member_idx].sort_values('start')

        # Check no gaps
        for i in range(len(member_data) - 1):
            assert member_data.iloc[i]['end'] == member_data.iloc[i + 1]['start'], \
                f"Gap detected for member {member_idx}"

        # Check full day coverage
        assert member_data.iloc[0]['start'] == 0, f"Member {member_idx} doesn't start at 0"
        assert member_data.iloc[-1]['end'] == 1440, f"Member {member_idx} doesn't end at 1440"

        print(f"  ✓ Member {member_idx}: continuous time coverage (0-1440 min)")

    print("\n" + "="*80)
    print("TEST PASSED: Encode-decode roundtrip successful!")
    print("="*80 + "\n")
