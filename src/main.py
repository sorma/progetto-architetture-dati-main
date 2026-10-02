import argparse
import sys
import os

# Aggiungo src al path per gli import relativi
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description="Sistema Unificato RAG: Optimization & Generation")
    parser.add_argument("--phase", choices=["optimization", "generation", "all", "combined-graphs"], required=True, help="Seleziona la fase da eseguire")
    
    # Recupero eventuali argomenti extra da passare ai moduli
    args, unknown = parser.parse_known_args()

    if args.phase in ["optimization", "all"]:
        print("\n" + "="*60)
        print("  AVVIO FASE DI OTTIMIZZAZIONE RAG")
        print("="*60)
        
        # Filtro sys.argv per non passare l'argomento "--phase"
        sys.argv = [sys.argv[0]] + unknown
        
        from optimization_phase.main import main as opt_main
        opt_main()

    if args.phase in ["generation", "all"]:
        print("\n" + "="*60)
        print("  AVVIO FASE GENERATIVA RAG")
        print("="*60)
        
        sys.argv = [sys.argv[0]] + unknown
        
        from generative_phase.main import main as gen_main
        gen_main()

    if args.phase == "combined-graphs":
        print("\n" + "="*60)
        print("  AVVIO GENERAZIONE GRAFICI CONGIUNTI")
        print("="*60)
        from combined_analysis import genera_tutti_grafici_congiunti
        genera_tutti_grafici_congiunti()

if __name__ == "__main__":
    main()
