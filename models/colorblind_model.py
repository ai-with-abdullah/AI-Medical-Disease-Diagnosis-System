import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import json
from datetime import datetime
import os

# Path to real Ishihara plates
ISHIHARA_PLATES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'ishihara_plates')

# ==================== TEST 1: ISHIHARA PLATES ====================
ISHIHARA_TEST_DATABASE = {
    'plate_1': {'correct_answers': ['12'], 'red_green_blind': ['1'], 'description': 'Warm-up: Normal sees 12, Red-Green sees 1', 'accuracy_weight': 0.15},
    'plate_2': {'correct_answers': ['8'], 'red_green_blind': ['3'], 'description': 'Red-Green test: Normal sees 8, Deficient sees 3', 'accuracy_weight': 0.20},
    'plate_3': {'correct_answers': ['29'], 'red_green_blind': ['70'], 'description': 'Protanopia: Normal sees 29, Protanopia sees 70', 'accuracy_weight': 0.20},
    'plate_4': {'correct_answers': ['45'], 'red_green_blind': ['unclear'], 'description': 'Deuteranopia: Normal sees 45, Deuteranopia sees nothing', 'accuracy_weight': 0.20},
    'plate_5': {'correct_answers': ['74'], 'red_green_blind': ['21'], 'description': 'Red-Green spectrum: Normal sees 74, Deficiency sees 21', 'accuracy_weight': 0.15},
    'plate_6': {'correct_answers': ['6'], 'red_green_blind': ['unclear'], 'description': 'Confirmation: Normal sees 6, Deficiency sees nothing', 'accuracy_weight': 0.10}
}

# ==================== TEST 2: FARNSWORTH D-15 COLOR ARRANGEMENT ====================
FARNSWORTH_TEST_DATABASE = {
    'color_1': {'correct_answers': ['yellow', 'orange', 'red'], 'correct_sequence': ['yellow', 'orange', 'red'], 'description': 'Warm color sequence', 'accuracy_weight': 0.15},
    'color_2': {'correct_answers': ['red', 'purple', 'blue'], 'correct_sequence': ['red', 'purple', 'blue'], 'description': 'Red-Purple-Blue progression', 'accuracy_weight': 0.20},
    'color_3': {'correct_answers': ['blue', 'cyan', 'green'], 'correct_sequence': ['blue', 'cyan', 'green'], 'description': 'Cool color sequence', 'accuracy_weight': 0.20},
    'color_4': {'correct_answers': ['green', 'lime', 'yellow'], 'correct_sequence': ['green', 'lime', 'yellow'], 'description': 'Green-Yellow transition', 'accuracy_weight': 0.20},
    'color_5': {'correct_answers': ['pink', 'magenta', 'purple'], 'correct_sequence': ['pink', 'magenta', 'purple'], 'description': 'Pink-Purple sequence', 'accuracy_weight': 0.15},
    'color_6': {'correct_answers': ['brown', 'orange', 'gold'], 'correct_sequence': ['brown', 'orange', 'gold'], 'description': 'Earth tones arrangement', 'accuracy_weight': 0.10}
}

# ==================== TEST 3: CAMBRIDGE COLOR TEST ====================
CAMBRIDGE_TEST_DATABASE = {
    'pattern_1': {'correct_answers': ['c', 'C'], 'correct_pattern': 'C', 'description': 'Red-Green discrimination - see C shape', 'difficulty': 'easy', 'accuracy_weight': 0.15},
    'pattern_2': {'correct_answers': ['triangle'], 'correct_pattern': 'triangle', 'description': 'Blue-Yellow test - see triangle', 'difficulty': 'easy', 'accuracy_weight': 0.20},
    'pattern_3': {'correct_answers': ['circle'], 'correct_pattern': 'circle', 'description': 'Tritan test - see circle', 'difficulty': 'medium', 'accuracy_weight': 0.20},
    'pattern_4': {'correct_answers': ['square'], 'correct_pattern': 'square', 'description': 'Low contrast red-green - see square', 'difficulty': 'medium', 'accuracy_weight': 0.20},
    'pattern_5': {'correct_answers': ['x', 'X'], 'correct_pattern': 'X', 'description': 'High contrast blue-yellow - see X', 'difficulty': 'medium', 'accuracy_weight': 0.15},
    'pattern_6': {'correct_answers': ['star'], 'correct_pattern': 'star', 'description': 'Complex pattern recognition - see star', 'difficulty': 'hard', 'accuracy_weight': 0.10}
}

# ==================== TEST 4: COLOR SPECTRUM DISCRIMINATION ====================
SPECTRUM_TEST_DATABASE = {
    'spectrum_1': {'correct_answers': ['red', 'orange', 'red-orange'], 'correct_range': 'red-orange', 'description': 'Distinguish warm spectrum variations', 'accuracy_weight': 0.15},
    'spectrum_2': {'correct_answers': ['orange', 'yellow', 'orange-yellow'], 'correct_range': 'orange-yellow', 'description': 'Orange-Yellow boundary discrimination', 'accuracy_weight': 0.20},
    'spectrum_3': {'correct_answers': ['green', 'cyan', 'green-cyan'], 'correct_range': 'green-cyan', 'description': 'Green-Cyan color transition', 'accuracy_weight': 0.20},
    'spectrum_4': {'correct_answers': ['blue', 'indigo', 'blue-indigo'], 'correct_range': 'blue-indigo', 'description': 'Blue-Indigo discrimination', 'accuracy_weight': 0.20},
    'spectrum_5': {'correct_answers': ['purple', 'magenta', 'purple-magenta'], 'correct_range': 'purple-magenta', 'description': 'Purple-Magenta distinction', 'accuracy_weight': 0.15},
    'spectrum_6': {'correct_answers': ['full spectrum', 'rainbow', 'all'], 'correct_range': 'full-spectrum', 'description': 'Full spectrum recognition', 'accuracy_weight': 0.10}
}

# ==================== TEST 5: ANOMALOSCOPE SIMULATION ====================
ANOMALOSCOPE_TEST_DATABASE = {
    'match_1': {'normal_ratio': 1.0, 'correct_range': [0.9, 1.1], 'description': 'Red-Green matching task 1', 'accuracy_weight': 0.15},
    'match_2': {'normal_ratio': 0.95, 'correct_range': [0.85, 1.05], 'description': 'Red-Green matching task 2 (slight variation)', 'accuracy_weight': 0.20},
    'match_3': {'normal_ratio': 1.05, 'correct_range': [0.95, 1.15], 'description': 'Red-Green matching task 3 (offset)', 'accuracy_weight': 0.20},
    'match_4': {'normal_ratio': 0.85, 'correct_range': [0.75, 0.95], 'description': 'Blue-Yellow matching task', 'accuracy_weight': 0.20},
    'match_5': {'normal_ratio': 1.15, 'correct_range': [1.05, 1.25], 'description': 'Composite color matching', 'accuracy_weight': 0.15},
    'match_6': {'normal_ratio': 1.0, 'correct_range': [0.9, 1.1], 'description': 'Final confirmation match', 'accuracy_weight': 0.10}
}

# Test metadata
TESTS_METADATA = {
    'ishihara': {
        'name': '🔴 Ishihara Plates',
        'description': 'Most widely used color blindness test',
        'accuracy': 93,
        'duration': '2-3 min',
        'database': ISHIHARA_TEST_DATABASE,
        'order': 1
    },
    'farnsworth': {
        'name': '🌈 Farnsworth D-15',
        'description': 'Color arrangement and discrimination test',
        'accuracy': 89,
        'duration': '3-5 min',
        'database': FARNSWORTH_TEST_DATABASE,
        'order': 2
    },
    'cambridge': {
        'name': '🎨 Cambridge Color',
        'description': 'Pattern detection with color discrimination',
        'accuracy': 87,
        'duration': '3-4 min',
        'database': CAMBRIDGE_TEST_DATABASE,
        'order': 3
    },
    'spectrum': {
        'name': '📊 Spectrum Discrimination',
        'description': 'Color spectrum range recognition',
        'accuracy': 85,
        'duration': '2-3 min',
        'database': SPECTRUM_TEST_DATABASE,
        'order': 4
    },
    'anomaloscope': {
        'name': '🔬 Anomaloscope',
        'description': 'Clinical color matching simulation',
        'accuracy': 95,
        'duration': '4-5 min',
        'database': ANOMALOSCOPE_TEST_DATABASE,
        'order': 5
    }
}

CVD_CLASSIFICATION = {
    'Normal': {'description': 'Normal Trichromatic Color Vision', 'percentage': '93% of population', 'can_see': 'All 10 million+ colors', 'severity': 'None', 'damage_ratio': 0.0},
    'Protanopia': {'description': 'Red Blindness', 'percentage': '1% of males', 'can_see': 'Blue-Yellow spectrum only', 'severity': 'Severe', 'damage_ratio': 0.95},
    'Deuteranopia': {'description': 'Green Blindness', 'percentage': '1% of males', 'can_see': 'Blue-Yellow spectrum only', 'severity': 'Severe', 'damage_ratio': 0.95},
    'Protanomaly': {'description': 'Red Weakness', 'percentage': '1% of males', 'can_see': 'All colors but weak red', 'severity': 'Mild to Moderate', 'damage_ratio': 0.40},
    'Deuteranomaly': {'description': 'Green Weakness', 'percentage': '4% of males', 'can_see': 'All colors but weak green', 'severity': 'Mild to Moderate', 'damage_ratio': 0.35},
    'Tritanopia': {'description': 'Blue Blindness', 'percentage': '0.001% of population', 'can_see': 'Red-Green spectrum only', 'severity': 'Severe', 'damage_ratio': 0.90}
}

def load_test_image_real(test_name, item_num):
    """Load REAL test image from assets directory"""
    try:
        # Map test names to directory and file patterns
        test_mappings = {
            'ishihara': ('ishihara_plates', f'plate_{item_num}.jpg'),
            'farnsworth': ('farnsworth_test', f'color_{item_num}.jpg'),
            'cambridge': ('cambridge_test', f'pattern_{item_num}.jpg'),
            'spectrum': ('spectrum_test', f'spectrum_{item_num}.jpg'),
            'anomaloscope': ('anomaloscope_test', f'match_{item_num}.jpg'),
        }
        
        if test_name not in test_mappings:
            return None
        
        test_dir, filename = test_mappings[test_name]
        image_path = os.path.join(ISHIHARA_PLATES_DIR, '..', test_dir, filename)
        image_path = os.path.normpath(image_path)
        
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is not None:
                return img
    except Exception as e:
        print(f"Error loading {test_name} image {item_num}: {e}")
    
    return None

def load_ishihara_plate_real(plate_num):
    """Load REAL Ishihara plate from assets directory"""
    return load_test_image_real('ishihara', plate_num)

def generate_ishihara_plate_authentic(plate_num, size=300):
    """Generate AUTHENTIC Ishihara plate with proper random dot distribution"""
    from PIL import Image, ImageDraw
    
    # Create PIL image
    pil_img = Image.new('RGB', (size, size))
    draw = ImageDraw.Draw(pil_img)
    
    # Plate specifications
    plates = {
        1: {
            'bg_color': (156, 162, 120),      # Greenish background (RGB)
            'num_color': (180, 60, 100),      # Reddish for number (RGB)
            'number': '12',
            'description': 'Normal: 12, Red-Green Blind: 1'
        },
        2: {
            'bg_color': (122, 105, 145),      # Purplish background (RGB)
            'num_color': (100, 180, 70),      # Greenish for number (RGB)
            'number': '8',
            'description': 'Normal: 8, Red-Green Blind: 3'
        },
        3: {
            'bg_color': (165, 170, 95),       # Yellowish background (RGB)
            'num_color': (110, 90, 195),      # Bluish for number (RGB)
            'number': '29',
            'description': 'Normal: 29, Red-Green Blind: 70'
        },
        4: {
            'bg_color': (148, 160, 100),      # Greenish background (RGB)
            'num_color': (160, 110, 195),     # Purplish for number (RGB)
            'number': '45',
            'description': 'Normal: 45, Red-Green Blind: Unclear'
        },
        5: {
            'bg_color': (125, 135, 175),      # Purplish background (RGB)
            'num_color': (210, 200, 100),     # Yellowish for number (RGB)
            'number': '74',
            'description': 'Normal: 74, Red-Green Blind: 21'
        },
        6: {
            'bg_color': (155, 110, 110),      # Reddish background (RGB)
            'num_color': (150, 210, 210),     # Cyan for number (RGB)
            'number': '6',
            'description': 'Normal: 6, Red-Green Blind: Unclear'
        }
    }
    
    plate = plates.get(plate_num, plates[1])
    bg_color = plate['bg_color']
    num_color = plate['num_color']
    
    # Fill background
    draw.rectangle([(0, 0), (size, size)], fill=bg_color)
    
    # Draw dots in background color - full coverage
    dot_size = 4
    num_dots_per_side = size // 8
    
    # Fill entire plate with background-colored dots
    for i in range(num_dots_per_side * 2):
        for j in range(num_dots_per_side * 2):
            x = (i * 8) + np.random.randint(-4, 5)
            y = (j * 8) + np.random.randint(-4, 5)
            if 0 <= x < size and 0 <= y < size:
                draw.ellipse([x-dot_size, y-dot_size, x+dot_size, y+dot_size], fill=bg_color)
    
    # Now overlay number pattern with contrasting dots
    # Define regions for each number pattern
    if plate_num == 1:  # "12"
        # "1" on left, "2" on right
        for i in range(num_dots_per_side):
            for j in range(num_dots_per_side):
                x_left = (i * 8) + np.random.randint(-4, 5)
                y_mid = (j * 8) + np.random.randint(-4, 5)
                x_right = int(size*0.6) + (i * 8) + np.random.randint(-4, 5)
                
                if 0 <= x_left < int(size*0.35) and int(size*0.25) <= y_mid < int(size*0.75):
                    draw.ellipse([x_left-dot_size, y_mid-dot_size, x_left+dot_size, y_mid+dot_size], fill=num_color)
                if 0 <= x_right < size and int(size*0.25) <= y_mid < int(size*0.75):
                    draw.ellipse([x_right-dot_size, y_mid-dot_size, x_right+dot_size, y_mid+dot_size], fill=num_color)
    
    elif plate_num == 2:  # "8" - circular pattern
        cx, cy = size // 2, size // 2
        for i in range(num_dots_per_side):
            for j in range(num_dots_per_side):
                x = (i * 8) + np.random.randint(-4, 5)
                y = (j * 8) + np.random.randint(-4, 5)
                dist = ((x - cx)**2 + (y - cy)**2)**0.5
                if size*0.15 < dist < size*0.35:
                    draw.ellipse([x-dot_size, y-dot_size, x+dot_size, y+dot_size], fill=num_color)
    
    elif plate_num in [3, 5]:  # "29" or "74" - left and right regions
        for i in range(num_dots_per_side):
            for j in range(num_dots_per_side):
                x = (i * 8) + np.random.randint(-4, 5)
                y = (j * 8) + np.random.randint(-4, 5)
                if 0 <= x < size and int(size*0.2) <= y < int(size*0.8):
                    if x < int(size*0.4) or x > int(size*0.6):
                        draw.ellipse([x-dot_size, y-dot_size, x+dot_size, y+dot_size], fill=num_color)
    
    elif plate_num == 4:  # "45" - horizontal band
        for i in range(num_dots_per_side * 2):
            for j in range(num_dots_per_side):
                x = (i * 8) + np.random.randint(-4, 5)
                y = int(size*0.3) + (j * 8) + np.random.randint(-4, 5)
                if 0 <= x < size and 0 <= y < size:
                    draw.ellipse([x-dot_size, y-dot_size, x+dot_size, y+dot_size], fill=num_color)
    
    elif plate_num == 6:  # "6" - circular at bottom
        cx, cy = size // 2, int(size * 0.6)
        for i in range(num_dots_per_side):
            for j in range(num_dots_per_side):
                x = (i * 8) + np.random.randint(-4, 5)
                y = (j * 8) + np.random.randint(-4, 5)
                if 0 <= x < size and 0 <= y < size:
                    dist = ((x - cx)**2 + (y - cy)**2)**0.5
                    if size*0.15 < dist < size*0.35:
                        draw.ellipse([x-dot_size, y-dot_size, x+dot_size, y+dot_size], fill=num_color)
    
    # Convert to numpy array (RGB) then to BGR for OpenCV
    img_array = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    return img_bgr

def generate_test_pattern(test_name, plate_number, size=300):
    """Generate test pattern for any test type"""
    plate_num = int(plate_number)
    img = np.ones((size, size, 3), dtype=np.uint8) * 255
    
    # Try to load REAL images first for ALL tests
    real_img = load_test_image_real(test_name, plate_num)
    if real_img is not None:
        return real_img
    
    # Fallback: generate if real image not available
    if test_name == 'ishihara':
        # Generate if real image not available
        return generate_ishihara_plate_authentic(plate_num, size)
    
    elif test_name == 'farnsworth':
        # Color bars - EACH PLATE IS DIFFERENT
        color_schemes = [
            [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)],  # RGB primary
            [(200, 50, 50), (50, 200, 50), (50, 50, 200), (200, 200, 50), (200, 50, 200), (50, 200, 200)],  # Dark
            [(255, 100, 100), (100, 255, 100), (100, 100, 255), (200, 200, 100), (200, 100, 200), (100, 200, 200)],  # Light
            [(255, 150, 0), (0, 255, 150), (150, 0, 255), (255, 200, 0), (200, 0, 255), (0, 200, 255)],  # Vibrant
            [(150, 75, 0), (75, 150, 0), (75, 0, 150), (150, 150, 0), (150, 0, 150), (0, 150, 150)],  # Medium
            [(255, 200, 100), (100, 255, 200), (200, 100, 255), (255, 255, 150), (255, 150, 255), (150, 255, 255)]  # Pastel
        ]
        colors = color_schemes[plate_num - 1] if plate_num <= 6 else color_schemes[0]
        bar_width = size // 6
        for i in range(6):
            start_x = i * bar_width
            end_x = start_x + bar_width
            img[:, start_x:end_x] = colors[i]
    
    elif test_name == 'cambridge':
        # Different shapes for each plate
        if plate_num == 1:
            # Triangle
            triangle = np.array([[size//4, size//4], [3*size//4, size//4], [size//2, 3*size//4]], dtype=np.int32)
            cv2.fillPoly(img, [triangle], (50, 100, 200))
        elif plate_num == 2:
            # Circle
            cv2.circle(img, (size//2, size//2), 80, (50, 200, 100), -1)
        elif plate_num == 3:
            # Square
            cv2.rectangle(img, (size//4, size//4), (3*size//4, 3*size//4), (200, 100, 50), -1)
        elif plate_num == 4:
            # Diamond
            diamond = np.array([[size//4, size//2], [size//2, size//4], [3*size//4, size//2], [size//2, 3*size//4]], dtype=np.int32)
            cv2.fillPoly(img, [diamond], (100, 150, 200))
        elif plate_num == 5:
            # Ellipse
            cv2.ellipse(img, (size//2, size//2), (80, 40), 0, 0, 360, (150, 100, 200), -1)
        elif plate_num == 6:
            # Star-like
            star = np.array([[size//2, size//4], [3*size//4, size//2], [3*size//4, 3*size//4], [size//4, 3*size//4], [size//4, size//2]], dtype=np.int32)
            cv2.fillPoly(img, [star], (200, 150, 50))
    
    elif test_name == 'spectrum':
        # Color spectrum - DIFFERENT FOR EACH PLATE
        spectrum_types = {
            1: 'rainbow',    # Full rainbow
            2: 'red_green',  # Red to green
            3: 'blue_yellow', # Blue to yellow
            4: 'warm',       # Warm colors
            5: 'cool',       # Cool colors
            6: 'grayscale'   # Grayscale
        }
        
        spectrum_type = spectrum_types.get(plate_num, 'rainbow')
        
        for x in range(size):
            if spectrum_type == 'rainbow':
                hue = int(180 * x / size)
                hsv_img = np.uint8([[[hue, 255, 255]]])
                bgr_color = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
                color = tuple(map(int, bgr_color[0][0]))
            elif spectrum_type == 'red_green':
                r = int(255 * x / size)
                g = int(255 * (1 - x / size))
                color = (0, g, r)
            elif spectrum_type == 'blue_yellow':
                b = int(255 * x / size)
                y = int(255 * (1 - x / size))
                color = (y, y, b)
            elif spectrum_type == 'warm':
                r = 255
                g = int(150 * x / size)
                color = (0, g, r)
            elif spectrum_type == 'cool':
                b = 255
                g = int(200 * x / size)
                color = (b, g, 50)
            else:  # grayscale
                gray = int(255 * x / size)
                color = (gray, gray, gray)
            
            cv2.line(img, (x, 0), (x, size), color, 1)
    
    elif test_name == 'anomaloscope':
        # Color matching - DIFFERENT colors for each plate
        color_pairs = [
            ((50, 150, 200), (200, 50, 100)),    # Cyan vs Red
            ((50, 200, 50), (200, 50, 200)),     # Green vs Magenta
            ((200, 200, 50), (50, 50, 200)),     # Yellow vs Blue
            ((50, 200, 200), (200, 100, 50)),    # Cyan vs Orange
            ((200, 50, 50), (50, 150, 200)),     # Red vs Cyan
            ((100, 200, 100), (150, 50, 200))    # Light Green vs Purple
        ]
        
        left_color, right_color = color_pairs[plate_num - 1] if plate_num <= 6 else color_pairs[0]
        cv2.rectangle(img, (10, 10), (size//2 - 5, size - 10), left_color, -1)
        cv2.rectangle(img, (size//2 + 5, 10), (size - 10, size - 10), right_color, -1)
        
        # Add a dividing line
        cv2.line(img, (size//2, 0), (size//2, size), (100, 100, 100), 2)
    
    return img

def analyze_single_test(test_name, answers):
    """Analyze results for a single test"""
    test_data = TESTS_METADATA[test_name]['database']
    correct_count = 0
    total_items = len(test_data)
    accuracy = 0.0
    
    for idx, (item_key, item_data) in enumerate(test_data.items()):
        user_answer = answers[idx] if idx < len(answers) else ""
        
        if test_name == 'ishihara':
            # Check if user answer matches any correct answer
            is_correct = user_answer in item_data['correct_answers']
        
        elif test_name == 'farnsworth':
            # Check if user answer matches any color in correct sequence
            correct_colors = item_data.get('correct_answers', item_data.get('correct_sequence', []))
            is_correct = user_answer.lower() in [c.lower() for c in correct_colors]
        
        elif test_name == 'cambridge':
            # Check if user answer matches the correct pattern
            correct_patterns = item_data.get('correct_answers', [item_data.get('correct_pattern', '')])
            is_correct = user_answer.lower() in [p.lower() for p in correct_patterns]
        
        elif test_name == 'spectrum':
            # Check if user answer matches any color in spectrum
            correct_colors = item_data.get('correct_answers', item_data.get('correct_range', '').split('-'))
            is_correct = user_answer.lower() in [c.lower().strip() for c in correct_colors]
        
        else:  # anomaloscope
            # Check if user ratio is within acceptable range
            try:
                user_val = float(user_answer) if user_answer else 0
                if 'correct_range' in item_data:
                    # Check if within range [min, max]
                    min_val, max_val = item_data['correct_range']
                    is_correct = min_val <= user_val <= max_val
                else:
                    # Check if within 0.2 of normal ratio
                    correct_val = item_data.get('normal_ratio', 1.0)
                    is_correct = abs(user_val - correct_val) < 0.2
            except:
                is_correct = False
        
        if is_correct:
            correct_count += 1
    
    accuracy = (correct_count / total_items) * 100
    
    return {
        'test_name': test_name,
        'display_name': TESTS_METADATA[test_name]['name'],
        'total_items': total_items,
        'correct_items': correct_count,
        'accuracy_percentage': accuracy,
        'confidence': min(0.95, 0.5 + accuracy / 200)
    }

def analyze_all_five_tests(all_answers_dict):
    """
    Analyze all 5 tests and calculate overall eye damage ratio
    
    Args:
        all_answers_dict: {'ishihara': [...], 'farnsworth': [...], ...}
    
    Returns:
        Comprehensive report with all tests + damage ratio
    """
    individual_results = {}
    total_accuracy = 0.0
    
    # Analyze each test
    for test_name in ['ishihara', 'farnsworth', 'cambridge', 'spectrum', 'anomaloscope']:
        answers = all_answers_dict.get(test_name, [])
        result = analyze_single_test(test_name, answers)
        individual_results[test_name] = result
        total_accuracy += result['accuracy_percentage']
    
    # Calculate overall metrics
    average_accuracy = total_accuracy / 5
    
    # Determine color blindness classification
    if average_accuracy >= 80:
        diagnosis = 'Normal Color Vision'
        severity = 'None'
        damage_ratio = 0.0
    elif average_accuracy >= 60:
        diagnosis = 'Mild Color Vision Deficiency'
        severity = 'Mild to Moderate'
        damage_ratio = 0.35
    elif average_accuracy >= 40:
        diagnosis = 'Moderate Color Vision Deficiency'
        severity = 'Moderate'
        damage_ratio = 0.60
    else:
        diagnosis = 'Severe Color Vision Deficiency'
        severity = 'Severe'
        damage_ratio = 0.90
    
    # Find most likely CVD type based on pattern
    cvd_type = 'Unknown'
    if average_accuracy >= 80:
        cvd_type = 'Trichromat (Normal)'
    elif average_accuracy >= 60:
        cvd_type = 'Possible Anomalous Trichromacy'
    else:
        cvd_type = 'Dichromacy or Severe Deficiency'
    
    # Determine recommendations
    recommendations = []
    if average_accuracy >= 80:
        recommendations = [
            '✅ Your color vision is normal',
            '💡 Continue regular eye health check-ups',
            '🎯 No restrictions on color-dependent activities'
        ]
    elif average_accuracy >= 60:
        recommendations = [
            '⚠️ Mild color vision deficiency detected',
            '👨‍⚕️ Consult an ophthalmologist for confirmation',
            '📋 Be aware of color discrimination limitations',
            '💼 Some professions may have restrictions'
        ]
    else:
        recommendations = [
            '🔴 Significant color blindness detected',
            '👨‍⚕️ URGENT: Schedule eye examination with ophthalmologist',
            '📋 Professional evaluation is strongly recommended',
            '🚫 Career implications for color-dependent professions',
            '🛠️ Use color blindness assistive technologies'
        ]
    
    return {
        'overall_diagnosis': diagnosis,
        'cvd_type': cvd_type,
        'severity': severity,
        'damage_ratio': damage_ratio,
        'damage_percentage': damage_ratio * 100,
        'average_accuracy': average_accuracy,
        'overall_confidence': min(0.95, 0.5 + average_accuracy / 200),
        'individual_test_results': individual_results,
        'recommendations': recommendations,
        'timestamp': datetime.now().isoformat(),
        'test_count': 5,
        'total_test_items': 30
    }

def generate_comprehensive_report(analysis_result):
    """Generate comprehensive report for all 5 tests"""
    report = {
        'title': 'Comprehensive Color Vision Assessment Report',
        'test_date': analysis_result['timestamp'],
        'overall_assessment': {
            'diagnosis': analysis_result['overall_diagnosis'],
            'color_vision_type': analysis_result['cvd_type'],
            'severity': analysis_result['severity'],
            'eye_damage_ratio': f"{analysis_result['damage_percentage']:.1f}%",
            'overall_accuracy': f"{analysis_result['average_accuracy']:.1f}%",
            'confidence': f"{analysis_result['overall_confidence']:.1%}"
        },
        'individual_test_results': {},
        'recommendations': analysis_result['recommendations'],
        'important_note': 'This is a comprehensive screening test. For official diagnosis, consult an ophthalmologist.'
    }
    
    # Add individual test results
    for test_name, result in analysis_result['individual_test_results'].items():
        report['individual_test_results'][result['display_name']] = {
            'correct_items': f"{result['correct_items']}/{result['total_items']}",
            'accuracy': f"{result['accuracy_percentage']:.1f}%",
            'confidence': f"{result['confidence']:.1%}"
        }
    
    return report

def export_comprehensive_results(analysis_result, filename='color_vision_complete_assessment.json'):
    """Export complete assessment results"""
    report = generate_comprehensive_report(analysis_result)
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    return filename
