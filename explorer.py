"""
Análisis Exploratorio de Datos (EDA) del dataset PhysioNet EEG Motor Movement/Imagery.
Especializado en ritmos Mu (8-13 Hz) y Beta (14-30 Hz) para interfaces Cerebro-Computadora (BCI).
"""

import os
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

# Configuración inicial: evitar spam de logs en consola
mne.set_log_level('WARNING')

# Directorio de salida para figuras (REQUISITO ESTRICTO: Sin plt.show())
PLOT_DIR = Path("eda_plots")
PLOT_DIR.mkdir(exist_ok=True, parents=True)

def load_eeg_data(filepath: Path) -> mne.io.Raw:
    """
    Carga un archivo de datos EEG en formato EDF.
    
    Neurofisiología: Los datos base se recogen generalmente entre 160 Hz y
    contienen diferentes fluctuaciones. Requerir información de localizaciones 
    de los electrodos a través del montaje (10-05) es clave para tareas espaciales.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {filepath}")
    
    # preload=True es estrictamente necesario para poder aplicar filtros
    raw = mne.io.read_raw_edf(filepath, preload=True)
    
    # Estandarizar nombres de canales (Physionet suele agregar sufijos incompatibles con MNE)
    mne.datasets.eegbci.standardize(raw)
    
    # Asignar un montaje estándar del sistema 10-05 internacional
    # Funciona para conocer topográficamente dónde está cada señal
    try:
        montage = mne.channels.make_standard_montage('standard_1005')
        raw.set_montage(montage, match_case=False, on_missing='ignore')
    except Exception as e:
        print(f"Advertencia al asignar montaje: {e}")
        
    return raw

def apply_bandpass_filter(raw: mne.io.Raw, l_freq: float = 8.0, h_freq: float = 30.0) -> mne.io.Raw:
    """
    Aplica un filtro pasa-banda para aislar ritmos sensorimotores.
    
    Neurofisiología: En BCI motor,filtramos usualmente entre 8 y 30 Hz 
    ya que nos interesan el ritmo Mu (8-13 Hz) y Beta (14-30 Hz). Estos ritmos 
    sufren una desincronización (ERD - caída de amplitud) cuando el usuario 
    se mueve o imagina moverse.
    """
    filtered_data = raw.copy().filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin')
    return filtered_data

def extract_events_and_epochs(raw: mne.io.Raw) -> Tuple[mne.Epochs, Dict[str, int]]:
    """
    Extrae eventos y segmenta la señal EEG según las tareas realizadas.
    
    Neurofisiología: Evaluaremos desde 1 seg antes del evento hasta 4 segundos 
    después. El periodo previo sirve como línea base ("baseline") para comparar la
    caída de voltaje en las bandas específicas durante la ejecución.
    """
    # Mapeo oficial para Runs de Imaginación Motora (Run 4, 8, 12 son para Izquierda/Derecha)
    # T0: Rest | T1: Imaginar Puño Izq | T2: Imaginar Puño Der
    event_mapping = {'T0': 1, 'T1': 2, 'T2': 3}
    event_id_dict = {'Rest': 1, 'Left_Fist_Imag': 2, 'Right_Fist_Imag': 3}
    
    events, _ = mne.events_from_annotations(raw, event_id=event_mapping)
    
    tmin, tmax = -1.0, 4.0
    epochs = mne.Epochs(
        raw, 
        events, 
        event_id=event_id_dict,
        tmin=tmin, 
        tmax=tmax, 
        baseline=(tmin, 0), # Restamos la media del baseline a toda la época
        preload=True
    )
    
    return epochs, event_id_dict

def analyze_psd_motor_cortex(raw: mne.io.Raw, channels: list = ['C3', 'Cz', 'C4']):
    """
    Calcula y gráfica la PSD con método de Welch para los electrodos de la zona motora.
    """
    picks = [ch for ch in channels if ch in raw.ch_names]
    if not picks:
        return
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Welch promedia ventanas solapadas disminuyendo la varianza del espectro.
    spectrum = raw.compute_psd(method='welch', fmin=8, fmax=30, picks=picks, n_fft=256)
    
    spectrum.plot(axes=ax, average=False, show=False)
    ax.set_title("Densidad Espectral de Potencia (PSD) - Corteza Motora \n(Filtro 8-30 Hz C3/Cz/C4)")
    
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "psd_motor_cortex.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def generate_topomaps(raw: mne.io.Raw):
    """
    Genera mapas de densidad energética a nivel del escalpo.
    """
    # Calcular el espectro para ambas bandas simultáneamente
    spectrum = raw.compute_psd(method='welch', fmin=8, fmax=30, n_fft=256)
    
    # Estructura requerida de bandas para graficado de MNE:
    # Formato de MNE moderno para espectros de PSD: 
    # spectrum.plot_topomap(bands={'Nombre Banda': (l_freq, h_freq)})
    bands = {
        'Mu (8-13 Hz)': (8, 13),
        'Beta (14-30 Hz)': (14, 30)
    }
    
    fig = spectrum.plot_topomap(bands=bands, show=False, ch_type='eeg')
    
    # Usualmente plot_topomap modifica o devuelve la figura instanciada
    output_fig = plt.gcf() if fig is None else fig
    
    output_fig.savefig(PLOT_DIR / "energy_topomaps.png", dpi=300, bbox_inches='tight')
    plt.close(output_fig)

def plot_erps(epochs: mne.Epochs, channels: list = ['C3', 'Cz', 'C4']):
    """
    Grafica los Potenciales Evocados (ERPs). Al promediar varias pruebas de la misma clase,
    el ruido de fondo se cancela y resalta la deflexión/voltaje vinculado al evento.
    """
    picks = [ch for ch in channels if ch in epochs.ch_names]
    if not picks:
        return
        
    # Extraemos evocaciones promedios y agrupamos
    evokeds = {
        'Imaginar Puño Izq': epochs['Left_Fist_Imag'].average(picks=picks),
        'Imaginar Puño Der': epochs['Right_Fist_Imag'].average(picks=picks)
    }
    
    # Crea comparativas directas superpuestas en el tiempo
    figs = mne.viz.plot_compare_evokeds(
        evokeds, 
        picks=picks, 
        title="Promedio ERP: Imaginación de Mano Izquierda vs Derecha",
        show=False,
        combine='mean' # Resumir todos los canales motores elegidos 
    )
    
    # Guardar manejo de Figuras para lista o instancia única
    if isinstance(figs, list):
        for i, fig in enumerate(figs):
            fig.savefig(PLOT_DIR / f"erp_comparison_{i}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
    else:
        figs.savefig(PLOT_DIR / "erp_comparison.png", dpi=300, bbox_inches='tight')
        plt.close(figs)

def main():
    try:
        # Base de datos del espacio de trabajo
        dataset_path = Path("/Users/danielsarmiento/Desktop/brazil/1er semestre/datasciense/physionet.org/files/eegmmidb/1.0.0/")
        # R04 corresponde a Motor Imagery (Imagina abrir/cerrar puño izquierdo o derecho)
        test_file = dataset_path / "S001" / "S001R04.edf" 
        
        print("1. Cargando datos EDF y asimilando base espacial (montaje)...")
        raw = load_eeg_data(test_file)
        
        print("2. Aplicando pre-procesamiento pasa-banda (8 - 30 Hz)...")
        raw_filtered = apply_bandpass_filter(raw)
        
        print("3. Ejecutando Análisis Espectral de Densidad de Potencia (Welch PSD)...")
        analyze_psd_motor_cortex(raw_filtered)
        
        print("4. Mapeando topografía espectral (Mu vs Beta)...")
        generate_topomaps(raw_filtered)
        
        print("5. Aislando Épocas y promediando Potenciales Evocados (ERPs)...")
        epochs, _ = extract_events_and_epochs(raw_filtered)
        plot_erps(epochs)
        
        print(f"\n¡Análisis Bioseñal exitoso! Gráficas generadas alojadas en: {PLOT_DIR.absolute()}")
        
    except Exception as e:
        print(f"\n[Error de Análisis] Ha ocurrido una anomalía durante el procesamiento: {e}")

if __name__ == '__main__':
    main()
