"""
test_memory_optimization.py
----------------------------
Tests spécifiques pour l'optimisation mémoire.
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_processing import optimize_memory


class TestMemoryOptimization:
    """Tests pour l'optimisation mémoire."""

    def test_memory_reduction_percentage(self):
        """Vérifie que la réduction mémoire est d'au moins 30%."""
        # Créer un DataFrame avec des types non optimisés
        df_test = pd.DataFrame({
            'int64_1': np.random.randint(0, 100, 10000, dtype=np.int64),
            'int64_2': np.random.randint(0, 50, 10000, dtype=np.int64),
            'float64_1': np.random.random(10000),
            'float64_2': np.random.random(10000) * 100,
            'object_col': ['A'] * 10000
        })
        
        mem_before = df_test.memory_usage(deep=True).sum() / 1024**2  # MB
        df_opt = optimize_memory(df_test)
        mem_after = df_opt.memory_usage(deep=True).sum() / 1024**2  # MB
        
        reduction = (mem_before - mem_after) / mem_before * 100
        
        print(f"\n📊 Mémoire avant: {mem_before:.2f} MB")
        print(f"   Mémoire après: {mem_after:.2f} MB")
        print(f"   Réduction: {reduction:.1f}%")
        
        assert reduction > 30, f"❌ Réduction trop faible: {reduction:.1f}%"
        print("✅ test_memory_reduction_percentage passé")

    def test_data_integrity_after_optimization(self):
        """Vérifie que les données restent identiques après optimisation."""
        df_test = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [1.1, 2.2, 3.3, 4.4, 5.5],
            'C': ['x', 'y', 'z', 'x', 'y']
        })
        
        df_opt = optimize_memory(df_test.copy())
        
        # Vérifier que les valeurs n'ont pas changé
        pd.testing.assert_frame_equal(df_test, df_opt, check_dtype=False)
        print("✅ test_data_integrity_after_optimization passé")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧠 TESTS D'OPTIMISATION MÉMOIRE")
    print("="*60)
    
    test = TestMemoryOptimization()
    test.test_memory_reduction_percentage()
    test.test_data_integrity_after_optimization()
    
    print("\n" + "="*60)
    print("✅ TESTS MÉMOIRE TERMINÉS")
    print("="*60)