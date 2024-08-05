from hezar.models import Model
from hezar.preprocessors import ImageProcessor
from hezar.trainer import Trainer, TrainerConfig
import pandas as  pd
from hezar.data import OCRDataset, OCRDatasetConfig
class PersianOCRDataset(OCRDataset):
    def __init__(self, config: OCRDatasetConfig, preprocessor, split=None, **kwargs):
        super().__init__(config=config, split=split, **kwargs)
        self.image_processor = preprocessor  # اضافه کردن preprocessor به کلاس
    def _load(self, split=None):
        # Load a dataframe here and make sure the split is fetched
        data = pd.read_excel(self.config.path)
        # preprocess if needed
        return data
    def __getitem__(self, index):
        path, text = self.data.iloc[index].values  # Corrected this line
        pixel_values = self.image_processor(path, return_tensors="torch")["pixel_values"][0]
        labels = self._text_to_tensor(text)
        inputs = {
            "pixel_values": pixel_values,
            "labels": labels,
        }
        return inputs
dataset_config = OCRDatasetConfig(
    path="/content/fine_tune_valid.xlsx",
    text_split_type="char_split",
    text_column="label",
    images_paths_column="image_path",
    reverse_digits=True,
)
model_id = "hezarai/crnn-fa-printed-96-long"
model = Model.load(model_id)
preprocessor = ImageProcessor.load(model_id)
train_dataset = PersianOCRDataset(dataset_config, preprocessor, split="train")
eval_dataset = PersianOCRDataset(dataset_config, preprocessor, split="test")
train_config = TrainerConfig(
    output_dir="crnn-fa-handwritten",
    task="image2text",
    device="cuda",
    batch_size=8,
    num_epochs=20,
    metrics=["cer"],
    metric_for_best_model="cer"
)
trainer = Trainer(
    config=train_config,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=train_dataset.data_collator,
    preprocessor=preprocessor,
)
trainer.train()
