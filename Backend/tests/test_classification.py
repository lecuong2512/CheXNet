import ast
from pathlib import Path
import re
import unittest

from Backend.ai_service.classification import DISEASE_LABELS, classification_summary

ROOT = Path(__file__).resolve().parents[2]


class ClassificationTests(unittest.TestCase):
    def test_catalogue_matches_readme_frontend_and_training_channel_order(self):
        readme = (ROOT / 'README.md').read_text(encoding='utf-8')
        rows = re.findall(r'\|\s*\d+\s*\|\s*\*\*([^*]+)\*\*\s*\|\s*([^|]+)\|', readme)
        catalogue = {name.replace('Pleural Thickening', 'Pleural_Thickening'): label.strip() for name, label in rows}
        self.assertEqual(DISEASE_LABELS, catalogue)
        display = (ROOT / 'Frontend/src/utils/display.ts').read_text(encoding='utf-8').split('};', 1)[0]
        frontend = {quoted or bare: value for quoted, bare, value in re.findall(r"(?:'([^']+)'|(\w+)):\s*'([^']+)'", display)}
        self.assertEqual(frontend, catalogue)
        tree = ast.parse((ROOT / 'Models/read_data.py').read_text(encoding='utf-8-sig'))
        columns = next(ast.literal_eval(node.value) for node in ast.walk(tree) if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == 'DISEASE_COLUMNS' for target in node.targets))
        self.assertEqual(columns, list(catalogue))
        for relative in ['Models/TrainModel.py', 'Finetune/TrainModel_finetune.py', 'Backend/main.py', 'predict_single.py']:
            tree = ast.parse((ROOT / relative).read_text(encoding='utf-8-sig'))
            labels = next(ast.literal_eval(node.value) for node in ast.walk(tree) if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == 'CLASS_NAMES' for target in node.targets))
            with self.subTest(source=relative):
                self.assertEqual(labels, columns)

    def test_seventh_positive_is_kept_even_without_heatmap(self):
        names = list(DISEASE_LABELS)
        probabilities = [{'disease': name, 'probability': 0.9 - i * 0.02} for i, name in enumerate(names[1:8])]
        visualized = [item['disease'] for item in probabilities[:6]]
        result = classification_summary(probabilities, names, visualized=visualized)
        self.assertEqual(len(result['findings']), 7)
        self.assertIn('Pneumonia', result['withoutHeatmap'])
        self.assertEqual(result['findings'][-1]['disease'], 'Pneumonia')

    def test_thresholds_follow_channel_names_despite_probability_sorting(self):
        result = classification_summary([
            {'disease': 'Pneumonia', 'probability': 0.6},
            {'disease': 'No Finding', 'probability': 0.3},
        ], ['No Finding', 'Pneumonia'], thresholds=[0.2, 0.7])
        self.assertEqual(result['findings'], [])
        self.assertEqual(result['decisions'][0]['threshold'], 0.7)
        self.assertFalse(result['decisions'][0]['aboveThreshold'])
        self.assertEqual(result['thresholdSource'], 'artifact')

    def test_conflicting_normal_and_pathology_are_flagged_without_suppressing_scores(self):
        result = classification_summary([
            {'disease': 'No Finding', 'probability': 0.9},
            {'disease': 'Pneumonia', 'probability': 0.5},
        ], ['No Finding', 'Pneumonia'], visualized=['Pneumonia'])
        self.assertTrue(result['conflictingNoFinding'])
        self.assertEqual(result['findings'][0]['probability'], 0.5)
        self.assertEqual(len(result['decisions']), 2)

    def test_invalid_threshold_artifacts_fail_instead_of_using_wrong_decisions(self):
        for thresholds in ([0.5], [float('nan'), 0.5], [0.5, -0.2], [0.5, 1.1]):
            with self.subTest(thresholds=thresholds), self.assertRaises(ValueError):
                classification_summary([], ['No Finding', 'Pneumonia'], thresholds)

    def test_no_positives_does_not_declare_patient_normal(self):
        result = classification_summary([
            {'disease': 'No Finding', 'probability': 0.2},
            {'disease': 'Pneumonia', 'probability': 0.3},
        ], ['No Finding', 'Pneumonia'])
        self.assertEqual(result['findings'], [])
        self.assertIn('chưa xác nhận ảnh bình thường', result['summary'])


if __name__ == '__main__':
    unittest.main()
