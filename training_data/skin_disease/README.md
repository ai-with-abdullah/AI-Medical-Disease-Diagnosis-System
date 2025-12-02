# Legacy Folder - Moved to skin_cancer

This folder has been **renamed** to `skin_cancer` to better reflect the purpose of the skin cancer detection model.

## Please Use the New Location

All skin cancer datasets should now be placed in:

```
training_data/skin_cancer/
```

The training script supports backward compatibility and will check this folder if the new `skin_cancer` folder doesn't exist.

## New Multi-Dataset Support

The skin cancer detection system now supports 6 different datasets:

1. **HAM10000** - 10,015 images
2. **ISIC 2019** - 25,331 images
3. **ISIC 2020** - 33,126 images
4. **PAD-UFES-20** - 2,298 images
5. **Melanoma Binary** - 10,605 images
6. **Pre-organized/Custom** - Any size

See `training_data/skin_cancer/README.md` for full setup instructions.

## Training Command

```bash
python training_scripts/train_skin_model.py
```
