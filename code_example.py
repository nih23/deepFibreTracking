"""Just example code as explanation. Usable for testing."""
from src.data import HCPDataContainer
from src.dataset import StreamlineDataset, ConcatenatedDataset
from src.tracker import CSDTracker, DTITracker

def main():
    """Main method"""
    data = HCPDataContainer(100307)
    csd_sl = CSDTracker(data, random_seeds=True, seeds_count=10000)
    dti_sl = DTITracker(data, random_seeds=True, seeds_count=10000)
    csd_sl.track()
    dti_sl.track()
    data = data.crop().normalize()
    csd = StreamlineDataset(csd_sl, data)
    dti = StreamlineDataset(dti_sl, data)
    dataset = ConcatenatedDataset([csd, dti])

if __name__ == "__main__":
    main()
