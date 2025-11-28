import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
from datetime import datetime

class ModelTrainer:
    def __init__(self, model_type='pneumonia'):
        self.model_type = model_type
        self.training_history = []
        self.cv_results = []
        
    def five_dataset_training_strategy(self, datasets):
        print(f"=== 5-Dataset Training Strategy for {self.model_type.upper()} ===")
        print(f"Total datasets: {len(datasets)}")
        
        train_datasets = datasets[:3]
        test_datasets = datasets[3:5]
        
        print("\nPhase 1: Training on datasets 1-3")
        X_train = np.concatenate([ds['X'] for ds in train_datasets])
        y_train = np.concatenate([ds['y'] for ds in train_datasets])
        
        model = self.build_model()
        
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        self.training_history.append({
            'phase': 'initial_training',
            'datasets_used': [1, 2, 3],
            'final_accuracy': history.history['accuracy'][-1],
            'final_val_accuracy': history.history['val_accuracy'][-1]
        })
        
        print("\nPhase 2: Testing on datasets 4-5")
        for idx, test_ds in enumerate(test_datasets, start=4):
            X_test = test_ds['X']
            y_test = test_ds['y']
            
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
            print(f"Dataset {idx} - Test Accuracy: {test_acc:.4f}")
            
            self.training_history.append({
                'phase': 'testing',
                'dataset': idx,
                'accuracy': test_acc,
                'loss': test_loss
            })
        
        print("\nPhase 3: Retraining on all 5 datasets")
        X_all = np.concatenate([ds['X'] for ds in datasets])
        y_all = np.concatenate([ds['y'] for ds in datasets])
        
        final_model = self.build_model()
        final_history = final_model.fit(
            X_all, y_all,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        self.training_history.append({
            'phase': 'final_training',
            'datasets_used': [1, 2, 3, 4, 5],
            'final_accuracy': final_history.history['accuracy'][-1],
            'final_val_accuracy': final_history.history['val_accuracy'][-1]
        })
        
        return final_model, final_history
    
    def cross_validation_training(self, X, y, n_folds=5):
        print(f"\n=== 5-Fold Cross-Validation for {self.model_type.upper()} ===")
        
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
            print(f"\nTraining Fold {fold}/{n_folds}")
            
            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X[val_idx]
            y_val_fold = y[val_idx]
            
            model = self.build_model()
            
            history = model.fit(
                X_train_fold, y_train_fold,
                epochs=30,
                batch_size=32,
                validation_data=(X_val_fold, y_val_fold),
                verbose=0
            )
            
            val_loss, val_acc = model.evaluate(X_val_fold, y_val_fold, verbose=0)
            
            fold_results.append({
                'fold': fold,
                'val_accuracy': val_acc,
                'val_loss': val_loss,
                'train_accuracy': history.history['accuracy'][-1]
            })
            
            print(f"Fold {fold} - Val Accuracy: {val_acc:.4f}, Val Loss: {val_loss:.4f}")
        
        mean_accuracy = np.mean([r['val_accuracy'] for r in fold_results])
        std_accuracy = np.std([r['val_accuracy'] for r in fold_results])
        
        print(f"\n=== Cross-Validation Results ===")
        print(f"Mean Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")
        
        self.cv_results = fold_results
        
        return fold_results
    
    def build_model(self):
        if self.model_type == 'pneumonia':
            from models.pneumonia_model import build_pneumonia_classifier
            return build_pneumonia_classifier('resnet50', num_classes=2)
        elif self.model_type == 'skin':
            from models.skin_model import build_skin_classifier
            return build_skin_classifier('resnet50', num_classes=7)
        elif self.model_type == 'colorblind':
            from models.colorblind_model import build_colorblind_cnn
            return build_colorblind_cnn(num_classes=6)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        
        accuracy = accuracy_score(y_test_classes, y_pred_classes)
        precision = precision_score(y_test_classes, y_pred_classes, average='weighted')
        recall = recall_score(y_test_classes, y_pred_classes, average='weighted')
        f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')
        
        cm = confusion_matrix(y_test_classes, y_pred_classes)
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist()
        }
        
        return metrics
    
    def save_training_report(self, filename='training_report.json'):
        report = {
            'model_type': self.model_type,
            'timestamp': datetime.now().isoformat(),
            'training_history': self.training_history,
            'cv_results': self.cv_results
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nTraining report saved to {filename}")
        
        return report

def generate_synthetic_dataset(num_samples=1000, image_shape=(224, 224, 3), num_classes=2):
    X = np.random.rand(num_samples, *image_shape).astype(np.float32)
    
    y = np.zeros((num_samples, num_classes))
    labels = np.random.randint(0, num_classes, num_samples)
    y[np.arange(num_samples), labels] = 1
    
    return {'X': X, 'y': y}

if __name__ == "__main__":
    print("=== Model Training Pipeline ===")
    print("\nThis script demonstrates the 5-dataset training strategy and cross-validation")
    print("For actual training, replace synthetic data with real medical datasets\n")
    
    print("Generating synthetic datasets for demonstration...")
    datasets = [generate_synthetic_dataset(num_samples=500) for _ in range(5)]
    
    trainer = ModelTrainer(model_type='pneumonia')
    
    print("\n" + "="*60)
    print("DEMO: Running 5-dataset training strategy")
    print("="*60)
    
    print("\nNote: For actual training with real datasets:")
    print("1. Load your 5 datasets from disk")
    print("2. Preprocess images (resize, normalize, augment)")
    print("3. Train on datasets 1-3")
    print("4. Validate on datasets 4-5")
    print("5. Retrain on all 5 datasets")
    print("6. Save the final model for deployment")
    
    print("\n" + "="*60)
    print("DEMO: 5-Fold Cross-Validation")
    print("="*60)
    
    X_all = np.concatenate([ds['X'] for ds in datasets])
    y_all = np.concatenate([ds['y'] for ds in datasets])
    
    print("\nNote: This demonstrates cross-validation methodology")
    print("Replace with your actual preprocessed dataset for real training")
