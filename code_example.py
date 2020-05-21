"""Just example code as explanation. Usable for testing."""
from src.data import HCPDataContainer
from src.data.postprocessing import res100, raw
from src.dataset import StreamlineDataset, ConcatenatedDataset, StreamlineClassificationDataset
from src.tracker import CSDTracker, DTITracker

def main():
    """Main method"""
    data = HCPDataContainer(100307)
    csd_sl = CSDTracker(data, random_seeds=True, seeds_count=10000)
    dti_sl = DTITracker(data, random_seeds=True, seeds_count=10000)
    csd_sl.track()
    dti_sl.track()
    data = data.crop().normalize()
    csd = StreamlineDataset(csd_sl, data, postprocessing=res100())
    dti = StreamlineDataset(dti_sl, data, postprocessing=res100())
    dataset = ConcatenatedDataset([csd, dti])
    csd_classification = StreamlineClassificationDataset(csd_sl, data, postprocessing=raw())
    dti_classification = StreamlineClassificationDataset(dti_sl, data, postprocessing=raw())
    dataset_classification = ConcatenatedDataset([csd_classification, dti_classification])

if __name__ == "__main__":
    main()
