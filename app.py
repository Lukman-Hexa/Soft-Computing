from flask import Flask, request, jsonify
import numpy as np
import time

app = Flask(__name__)

# ========================
# PSO - OPTIMASI PRODUKSI
# ========================

def objective_function(x):
    """Fungsi objektif: maksimalkan keuntungan"""
    x1, x2, x3 = x[0], x[1], x[2]
    profit = 50000 * x1 + 40000 * x2 + 30000 * x3
    
    # Penalty untuk constraint
    penalty = 0
    
    # Constraint 1: Waktu mesin <= 100 jam
    machine_time = 2 * x1 + 3 * x2 + 1 * x3
    if machine_time > 100:
        penalty += 10000
    
    # Constraint 2: Bahan baku <= 120 kg
    raw_material = 3 * x1 + 2 * x2 + 4 * x3
    if raw_material > 120:
        penalty += 10000
    
    # Constraint 3: Kapasitas gudang <= 50 unit
    total_units = x1 + x2 + x3
    if total_units > 50:
        penalty += 10000
    
    # Fitness = profit - penalty (karena kita ingin maximize)
    fitness = profit - penalty
    return fitness, machine_time, raw_material, total_units

class PSO:
    def __init__(self, n_particles, n_dimensions, bounds, w, c1, c2, max_iterations):
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
        self.bounds = bounds
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iterations = max_iterations
        
        # Inisialisasi partikel
        self.positions = np.random.uniform(bounds[0], bounds[1], (n_particles, n_dimensions))
        self.velocities = np.random.uniform(-1, 1, (n_particles, n_dimensions))
        self.personal_best_positions = self.positions.copy()
        self.personal_best_fitness = np.array([objective_function(pos)[0] for pos in self.positions])
        self.global_best_position = self.personal_best_positions[np.argmax(self.personal_best_fitness)]
        self.global_best_fitness = np.max(self.personal_best_fitness)
        
    def optimize(self):
        start_time = time.time()
        
        for iteration in range(self.max_iterations):
            for i in range(self.n_particles):
                # Update velocity
                r1 = np.random.random(self.n_dimensions)
                r2 = np.random.random(self.n_dimensions)
                
                cognitive = self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                social = self.c2 * r2 * (self.global_best_position - self.positions[i])
                
                self.velocities[i] = (self.w * self.velocities[i] + cognitive + social)
                
                # Update position
                self.positions[i] += self.velocities[i]
                
                # Batasi posisi dalam bounds
                self.positions[i] = np.clip(self.positions[i], self.bounds[0], self.bounds[1])
                
                # Evaluasi fitness
                fitness, _, _, _ = objective_function(self.positions[i])
                
                # Update personal best
                if fitness > self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best_positions[i] = self.positions[i].copy()
                    
                    # Update global best
                    if fitness > self.global_best_fitness:
                        self.global_best_fitness = fitness
                        self.global_best_position = self.positions[i].copy()
        
        execution_time = time.time() - start_time
        
        # Hitung detail constraint untuk solusi terbaik
        best_fitness, machine_time, raw_material, total_units = objective_function(self.global_best_position)
        
        return {
            'best_solution': self.global_best_position.tolist(),
            'max_profit': best_fitness,
            'machine_time_used': machine_time,
            'raw_material_used': raw_material,
            'total_units': total_units,
            'iterations': self.max_iterations,
            'execution_time': execution_time
        }

# ========================
# BAYESIAN NETWORK - DIAGNOSIS
# ========================

def calculate_bayesian_probability(age, smoking, symptoms):
    """
    Hitung P(Penyakit Jantung = Ya | Gejala)
    Menggunakan teorema Bayes: P(H|E) = P(E|H) * P(H) / P(E)
    """
    
    # Probabilitas prior
    p_merokok = {'ya': 0.30, 'tidak': 0.70}
    p_usia = {'muda': 0.60, 'tua': 0.40}
    
    # Probabilitas penyakit jantung berdasarkan merokok dan usia
    p_jantung_given_merokok_usia = {
        ('ya', 'tua'): 0.70,
        ('ya', 'muda'): 0.40,
        ('tidak', 'tua'): 0.25,
        ('tidak', 'muda'): 0.05
    }
    
    # Probabilitas gejala berdasarkan penyakit jantung
    p_gejala_given_jantung = {
        'nyeri_dada': {True: 0.80, False: 0.10},
        'sesak_napas': {True: 0.70, False: 0.15},
        'lelah': {True: 0.60, False: 0.30}
    }
    
    # Hitung P(Jantung = Ya) dan P(Jantung = Tidak)
    p_jantung_ya = p_jantung_given_merokok_usia[(smoking, age)]
    p_jantung_tidak = 1 - p_jantung_ya
    
    # Hitung P(Gejala | Jantung = Ya) dan P(Gejala | Jantung = Tidak)
    p_gejala_given_jantung_ya = 1.0
    p_gejala_given_jantung_tidak = 1.0
    
    for symptom in symptoms:
        p_gejala_given_jantung_ya *= p_gejala_given_jantung[symptom][True]
        p_gejala_given_jantung_tidak *= p_gejala_given_jantung[symptom][False]
    
    # Hitung P(Gejala) menggunakan hukum probabilitas total
    p_gejala = (p_gejala_given_jantung_ya * p_jantung_ya + 
                p_gejala_given_jantung_tidak * p_jantung_tidak)
    
    # Hitung P(Jantung = Ya | Gejala) menggunakan teorema Bayes
    if p_gejala == 0:
        p_jantung_ya_given_gejala = 0.0
    else:
        p_jantung_ya_given_gejala = (p_gejala_given_jantung_ya * p_jantung_ya) / p_gejala
    
    p_jantung_tidak_given_gejala = 1 - p_jantung_ya_given_gejala
    
    return {
        'probability_jantung_ya': p_jantung_ya_given_gejala,
        'probability_jantung_tidak': p_jantung_tidak_given_gejala,
        'patient_profile': {
            'age': age,
            'smoking': smoking,
            'symptoms': symptoms
        },
        'details': {
            'p_jantung_ya': p_jantung_ya,
            'p_jantung_tidak': p_jantung_tidak,
            'p_gejala_given_jantung_ya': p_gejala_given_jantung_ya,
            'p_gejala_given_jantung_tidak': p_gejala_given_jantung_tidak,
            'p_gejala': p_gejala
        }
    }

# ========================
# MARKOV CHAIN - PREDIKSI KREDIT
# ========================

def predict_credit_status(months, initial_distribution):
    """
    Prediksi status kredit nasabah menggunakan Markov Chain
    States: 0=Lancar, 1=Kurang Lancar, 2=Macet, 3=Write-off
    """
    start_time = time.time()
    
    # Matriks transisi P
    P = np.array([
        [0.85, 0.12, 0.03, 0.00],  # Lancar
        [0.40, 0.30, 0.30, 0.00],  # Kurang Lancar
        [0.00, 0.10, 0.70, 0.20],  # Macet
        [0.00, 0.00, 0.00, 1.00]   # Write-off
    ])
    
    # Distribusi awal
    current_distribution = np.array(initial_distribution)
    
    # Hitung distribusi setelah n bulan
    for _ in range(months):
        current_distribution = np.dot(current_distribution, P)
    
    execution_time = time.time() - start_time
    
    return {
        'months': months,
        'initial_distribution': initial_distribution,
        'final_distribution': current_distribution.tolist(),
        'execution_time': execution_time
    }

@app.route('/')
def index():
    return open('index.html').read()

@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        data = request.get_json()
        
        # Ambil parameter dari request
        n_particles = int(data['particles'])
        n_dimensions = int(data['dimensions'])
        bounds = data['bounds']
        w = float(data['w'])
        c1 = float(data['c1'])
        c2 = float(data['c2'])
        max_iterations = int(data['iterations'])
        
        # Jalankan PSO
        pso = PSO(n_particles, n_dimensions, bounds, w, c1, c2, max_iterations)
        result = pso.optimize()
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/diagnose', methods=['POST'])
def diagnose():
    try:
        data = request.get_json()
        
        age = data['age']  # 'muda' or 'tua'
        smoking = data['smoking']  # 'ya' or 'tidak'
        symptoms = data['symptoms']  # list of symptoms like ['nyeri_dada', 'sesak_napas']
        
        # Validasi input
        if age not in ['muda', 'tua']:
            return jsonify({'error': 'Usia harus "muda" atau "tua"'}), 400
        if smoking not in ['ya', 'tidak']:
            return jsonify({'error': 'Merokok harus "ya" atau "tidak"'}), 400
        valid_symptoms = ['nyeri_dada', 'sesak_napas', 'lelah']
        for sym in symptoms:
            if sym not in valid_symptoms:
                return jsonify({'error': f'Gejala tidak valid: {sym}'}), 400
        
        # Hitung probabilitas Bayesian
        result = calculate_bayesian_probability(age, smoking, symptoms)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_credit', methods=['POST'])
def predict_credit():
    try:
        data = request.get_json()
        
        months = int(data['months'])
        initial_distribution = data['initial_distribution']  # list of 4 probabilities
        
        # Validasi
        if months < 1:
            return jsonify({'error': 'Jumlah bulan harus minimal 1'}), 400
        if len(initial_distribution) != 4:
            return jsonify({'error': 'Distribusi awal harus memiliki 4 elemen'}), 400
        if abs(sum(initial_distribution) - 1.0) > 0.001:
            return jsonify({'error': 'Jumlah distribusi awal harus 1.0'}), 400
        
        # Prediksi Markov
        result = predict_credit_status(months, initial_distribution)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
