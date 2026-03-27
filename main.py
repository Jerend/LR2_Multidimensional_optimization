import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.optimize import minimize_scalar
import matplotlib.colors as colors
import csv
import os


class RosenbrockMethod:
    def __init__(self):
        self.current_function = None
        self.points_history = []
        self.all_steps_history = []
        self.search_interval = None
        self.functions = {
            "F₁(x) = (3x₁² + x₂)² + (2x₁ - 3x₂)²": self.function1,
            "F₂(x) = 9x₁² + 16x₂² - 90x₁ - 128x₂": self.function2,
            "F₃(x) = (x₁ - 2)⁴ + (x₁ - 2x₂)²": self.function3
        }
        
    def function1(self, x):
        return (3 * x[0]**2 + x[1])**2 + (2 * x[0] - 3 * x[1])**2
    
    def function2(self, x):
        return 9 * x[0]**2 + 16 * x[1]**2 - 90 * x[0] - 128 * x[1]
    
    def function3(self, x):
        return (x[0] - 2)**4 + (x[0] - 2 * x[1])**2
        
    def golden_section_search(self, x, direction, a, b, eps):
        """
        Метод золотого сечения для одномерной оптимизации
        x - текущая точка (список [x1, x2])
        direction - направление поиска (список [d1, d2])
        a, b - начальный интервал
        eps - точность поиска
        Возвращает оптимальное значение λ
        """
        phi = 0.618  # Коэффициент золотого сечения
        
        def objective(alpha):             # Вычисление точки: x + alpha * direction
            x_alpha = [
                x[0] + alpha * direction[0],
                x[1] + alpha * direction[1]
            ]
            return self.current_function(x_alpha)
        
        # Начальный этап
        lamda = a + (1 - phi) * (b - a)
        mu = a + phi * (b - a)
        
        f_lamda = objective(lamda)
        f_mu = objective(mu)
        
        # Основной цикл
        while b - a > eps:
            if f_lamda > f_mu:
                # Сужаем интервал справа
                a = lamda
                lamda = mu
                f_lamda = f_mu
                
                mu = a + phi * (b - a)
                f_mu = objective(mu)
            else:
                # Сужаем интервал слева
                b = mu
                mu = lamda
                f_mu = f_lamda
                
                lamda = a + (1 - phi) * (b - a)
                f_lamda = objective(lamda)
        
        # Возвращаем середину интервала
        return (a + b) / 2

    def run_algorithm(self, x_start, directions, eps, search_interval):
        """Запуск алгоритма Розенброка"""
        self.points_history.clear()
        self.all_steps_history.clear()
        
        # Начальный этап      
        xk = [x_start[0], x_start[1]]       # xk - текущая точка (начальная точка x₁)       
        d = [directions[0], directions[1]]  # d - направления [d₁, d₂]          
        k = 1                               # k - номер итерации
        table_data = []   
        
        while True:
            # ШАГ 1
            y = [xk[0], xk[1]]          
            lambda_values = [] # Список для хранения найденных λ
            self.all_steps_history.append([y[0], y[1]]) # Сохраняем начальную точку итерации
            
            for j in range(2):
                dj = d[j]  # Текущее направление (d1 или d2)                            
                a, b = search_interval # Получение интервала для поиска λ
           
                lambda_j = self.golden_section_search(y, dj, a, b, eps) # Находит оптимальный шаг λ методом золотого сечения
                lambda_values.append(lambda_j)
                
                # Вычисляем новую точку: y_{j+1} = y_j + λ_j * d_j
                y_next = [
                    y[0] + lambda_j * dj[0],
                    y[1] + lambda_j * dj[1]
                ]
                
                # Сохраняем точку
                self.all_steps_history.append([y_next[0], y_next[1]])
                
                # Формируем строку для таблицы
                if j == 0:  # Первое направление
                    row = {
                        'K': k,
                        'X_k': f'({xk[0]:.2f}, {xk[1]:.2f})',
                        'F(X_k)': f'{self.current_function(xk):.3f}',
                        'j': j + 1,
                        'y_j': f'({y[0]:.2f}, {y[1]:.2f})',
                        'f(y_j)': f'{self.current_function(y):.3f}',
                        'd_j': f'({dj[0]:.2f}, {dj[1]:.2f})',
                        'λj': f'{lambda_j:.2f}',
                        'y_j+1': f'({y_next[0]:.2f}, {y_next[1]:.2f})',
                        'f(y_j+1)': f'{self.current_function(y_next):.3f}'
                    }
                else:  # Второе направление
                    row = {
                        'K': '',
                        'X_k': '',
                        'F(X_k)': '',
                        'j': j + 1,
                        'y_j': f'({y[0]:.2f}, {y[1]:.2f})',
                        'f(y_j)': f'{self.current_function(y):.3f}',
                        'd_j': f'({dj[0]:.2f}, {dj[1]:.2f})',
                        'λj': f'{lambda_j:.2f}',
                        'y_j+1': f'({y_next[0]:.2f}, {y_next[1]:.2f})',
                        'f(y_j+1)': f'{self.current_function(y_next):.3f}'
                    }
                
                table_data.append(row)
                
                # Обновляем y для следующего направления
                y = [y_next[0], y_next[1]]
            
            # Шаг 2
            x_next = [y[0], y[1]] # x_{k+1} = y₃
            
            # Сохраняем информацию об итерации
            self.points_history.append((
                k, 
                [xk[0], xk[1]], 
                self.current_function(xk),
                [x_next[0], x_next[1]], 
                self.current_function(x_next)
            ))
                       
            # Критерий остановки
            if (((x_next[0] - xk[0])**2 + (x_next[1] - xk[1])**2)**0.5) < eps:
                break
            
            # Шаг 3
            # Процесс Грама-Шмидта        
            lambda_1 = lambda_values[0]  # λ₁
            lambda_2 = lambda_values[1]  # λ₂
            
            d1 = d[0]  # d₁
            d2 = d[1]  # d₂

            a1 = [
                lambda_1 * d1[0] + lambda_2 * d2[0],
                lambda_1 * d1[1] + lambda_2 * d2[1]
            ]
            a2 = [
                lambda_2 * d2[0],
                lambda_2 * d2[1]
            ]

            # Первое направление: b₁ = a₁ / ||a₁||
            norm_a1 = (a1[0]**2 + a1[1]**2)**0.5
            if norm_a1 > 0:
                b1 = [a1[0] / norm_a1, a1[1] / norm_a1]
            else:
                b1 = [a1[0], a1[1]]
            
            # Второе направление: b₂ = a₂ - (a₂·b₁)*b₁, затем нормируем
            dot_product = a2[0] * b1[0] + a2[1] * b1[1]
            b2 = [
                a2[0] - dot_product * b1[0],
                a2[1] - dot_product * b1[1]
            ]
            
            norm_b2 = (b2[0]**2 + b2[1]**2)**0.5
            if norm_b2 > 0:
                b2 = [b2[0] / norm_b2, b2[1] / norm_b2]
               
            d = [b1, b2]                # Обновляем направления для следующей итерации
            xk = [x_next[0], x_next[1]] # Обновляем текущую точку
            k += 1                      # Увеличиваем счетчик итераций
        
        return table_data

class RosenbrockGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Метод Розенброка")
        self.root.geometry("1400x800")
        
        self.rosenbrock = RosenbrockMethod()
        
        self.setup_ui()
        
    def setup_ui(self):
        # Создаем основной фрейм
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Создаем верхнюю панель (слева параметры, справа график)
        top_frame = ttk.Frame(main_frame)
        top_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Левая часть - параметры алгоритма
        input_frame = ttk.LabelFrame(top_frame, text="Параметры", padding="10")
        input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N), padx=(0, 10))
        
        # Выбор функции
        ttk.Label(input_frame, text="Выберите функцию:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.function_var = tk.StringVar()
        function_combo = ttk.Combobox(input_frame, textvariable=self.function_var, values=list(self.rosenbrock.functions.keys()), state="readonly", width=27)
        function_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        function_combo.current(0)
            
        # Начальная точка X1
        ttk.Label(input_frame, text="Начальная точка X1:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        
        # Фрейм для координат X1
        x1_frame = ttk.Frame(input_frame)
        x1_frame.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(x1_frame, text="x₁ =").grid(row=0, column=0, padx=2)
        self.x1_1 = tk.StringVar(value="0.0")
        ttk.Entry(x1_frame, textvariable=self.x1_1, width=8).grid(row=0, column=1, padx=2)
        
        ttk.Label(x1_frame, text="x₂ =").grid(row=0, column=2, padx=2)
        self.x1_2 = tk.StringVar(value="3.0")
        ttk.Entry(x1_frame, textvariable=self.x1_2, width=8).grid(row=0, column=3, padx=2)
        
        # Направление d1
        ttk.Label(input_frame, text="Направление d₁:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        
        # Фрейм для координат d1
        d1_frame = ttk.Frame(input_frame)
        d1_frame.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(d1_frame, text="d₁₁ =").grid(row=0, column=0, padx=2)
        self.d1_1 = tk.StringVar(value="1.0")
        ttk.Entry(d1_frame, textvariable=self.d1_1, width=8).grid(row=0, column=1, padx=2)
        
        ttk.Label(d1_frame, text="d₁₂ =").grid(row=0, column=2, padx=2)
        self.d1_2 = tk.StringVar(value="0.0")
        ttk.Entry(d1_frame, textvariable=self.d1_2, width=8).grid(row=0, column=3, padx=2)
        
        # Направление d2
        ttk.Label(input_frame, text="Направление d₂:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        
        # Фрейм для координат d2
        d2_frame = ttk.Frame(input_frame)
        d2_frame.grid(row=4, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(d2_frame, text="d₂₁ =").grid(row=0, column=0, padx=2)
        self.d2_1 = tk.StringVar(value="0.0")
        ttk.Entry(d2_frame, textvariable=self.d2_1, width=8).grid(row=0, column=1, padx=2)
        
        ttk.Label(d2_frame, text="d₂₂ =").grid(row=0, column=2, padx=2)
        self.d2_2 = tk.StringVar(value="1.0")
        ttk.Entry(d2_frame, textvariable=self.d2_2, width=8).grid(row=0, column=3, padx=2)
          
        # Точность
        ttk.Label(input_frame, text="Точность (ε):").grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        self.eps_var = tk.StringVar(value="0.2")
        ttk.Entry(input_frame, textvariable=self.eps_var, width=8).grid(row=5, column=1, sticky=tk.W, padx=5, pady=5)

        # Интервал для поиска λ
        ttk.Label(input_frame, text="Интервал для λ:").grid(row=6, column=0, sticky=tk.W, padx=5, pady=5)

        # Фрейм для интервала
        interval_frame = ttk.Frame(input_frame)
        interval_frame.grid(row=6, column=1, sticky=tk.W, padx=5, pady=5)

        ttk.Label(interval_frame, text="a =").grid(row=0, column=0, padx=2)
        self.interval_a = tk.StringVar(value="-10.0")
        ttk.Entry(interval_frame, textvariable=self.interval_a, width=8).grid(row=0, column=1, padx=2)

        ttk.Label(interval_frame, text="b =").grid(row=0, column=2, padx=2)
        self.interval_b = tk.StringVar(value="10.0")
        ttk.Entry(interval_frame, textvariable=self.interval_b, width=8).grid(row=0, column=3, padx=2)

        # Фрейм для кнопок
        buttons_frame = ttk.Frame(input_frame)
        buttons_frame.grid(row=7, column=0, columnspan=2, pady=10)
        
        # Кнопка запуска
        self.start_button = ttk.Button(buttons_frame, text="Запустить алгоритм", command=self.run_algorithm, width=20)
        self.start_button.grid(row=0, column=0, padx=5)
        
        # Кнопка сохранения
        self.save_button = ttk.Button(buttons_frame, text="Сохранить в CSV", command=self.save_to_csv, width=20)
        self.save_button.grid(row=0, column=1, padx=5)
        
        # Фрейм для вывода результата
        result_frame = ttk.LabelFrame(input_frame, text="Результат", padding="5")
        result_frame.grid(row=8, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        self.result_label = tk.Label(result_frame, text="Алгоритм не запущен", font=('Arial', 10), justify=tk.LEFT)
        self.result_label.grid(row=0, column=0, padx=5, pady=5)
        
        # Правая часть - график
        chart_frame = ttk.LabelFrame(top_frame, text="График", padding="10")
        chart_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Создаем график
        self.fig = Figure(figsize=(7, 3.5), dpi=100, facecolor='white')
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Нижняя часть - таблица с прокруткой
        table_frame = ttk.LabelFrame(main_frame, text="Результаты вычислений", padding="10")
        table_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Создаем фрейм для таблицы с прокруткой
        table_container = ttk.Frame(table_frame)
        table_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Создаем таблицу
        columns = ('K', 'X_k', 'F(X_k)', 'j', 'y_j', 'f(y_j)', 'd_j', 'λj', 'y_j+1', 'f(y_j+1)')
        self.tree = ttk.Treeview(table_container, columns=columns, show='headings')
        
        # Настраиваем заголовки
        column_widths = {'K': 50, 'X_k': 100, 'F(X_k)': 80, 'j': 40, 'y_j': 100, 'f(y_j)': 80, 'd_j': 100, 'λj': 60, 'y_j+1': 100, 'f(y_j+1)': 80}
        
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=column_widths.get(col, 80), anchor='center')
        
        # Добавляем скроллбары
        v_scrollbar = ttk.Scrollbar(table_container, orient=tk.VERTICAL, command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(table_container, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Размещаем таблицу и скроллбары
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Настройка весов для растягивания
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=0)
        main_frame.rowconfigure(1, weight=1)
        top_frame.columnconfigure(0, weight=0)
        top_frame.columnconfigure(1, weight=1)
        top_frame.rowconfigure(0, weight=1)
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        table_container.columnconfigure(0, weight=1)
        table_container.rowconfigure(0, weight=1)
    
    def update_result_display(self):
        """Обновление отображения результата"""
        if self.rosenbrock.all_steps_history:
            opt_point = self.rosenbrock.all_steps_history[-1]
            f_opt = self.rosenbrock.current_function(opt_point)
            
            result_text = f"Оптимальная точка: ({opt_point[0]:.6f}, {opt_point[1]:.6f})\nЗначение функции: {f_opt:.6f}"
            self.result_label.config(text=result_text, fg="black")
        else:
            self.result_label.config(text="Алгоритм не запущен", fg="gray")
    
    def plot_function_contour(self, ax, x_range, y_range, function):
        """Построение контурного графика функции"""
        x = np.linspace(x_range[0], x_range[1], 100)
        y = np.linspace(y_range[0], y_range[1], 100)
        X, Y = np.meshgrid(x, y)
        
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = function([X[i, j], Y[i, j]])
        
        # Контуры
        contour = ax.contour(X, Y, Z, levels=20, colors='black', linewidths=0.5, alpha=0.6)
             
        return X, Y, Z
    
    def get_adaptive_levels(self, Z, num_levels=15):
        """
        Автоматическое определение уровней для контурного графика
        Z - матрица значений функции
        num_levels - желаемое количество уровней
        """
        z_min = np.min(Z)
        z_max = np.max(Z)
        
        # Для функции с отрицательным минимумом (как F2)
        if z_min < 0:
            # Определяем интервал от минимума до нуля
            negative_range = abs(z_min)
            
            # Создаем уровни: несколько уровней вблизи минимума и несколько выше
            if negative_range > 0:
                # Логарифмические уровни для области вблизи минимума (отрицательные значения)
                # Добавляем уровни: минимум + небольшие значения
                near_min_levels = []
                for val in [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]:
                    level = z_min + val
                    if level < 0:
                        near_min_levels.append(level)
                
                # Положительные уровни (выше нуля)
                positive_levels = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
                positive_levels = [l for l in positive_levels if l <= z_max]
                
                levels = sorted(set(near_min_levels + positive_levels))
            else:
                levels = np.linspace(z_min, z_max, num_levels)
        else:
            # Для функции с положительным минимумом
            # Используем логарифмическую шкалу для лучшего отображения
            if z_min > 0:
                log_min = np.log10(max(z_min, 0.1))
                log_max = np.log10(z_max)
                levels = np.logspace(log_min, log_max, num_levels)
            else:
                levels = np.linspace(z_min, z_max, num_levels)
        
        # Фильтруем уровни, чтобы они были в диапазоне значений
        levels = [l for l in levels if z_min <= l <= z_max]
        
        # Добавляем уровни между min и max
        if len(levels) < 8:
            linear_levels = np.linspace(z_min, z_max, 12)
            levels = sorted(set(list(levels) + list(linear_levels)))
        
        return sorted(set(levels))
        
    def update_chart(self):
        """Обновление графика"""
        self.ax.clear()
        
        if not self.rosenbrock.all_steps_history:
            return
        
        # Получение выбранной функции
        function_name = self.function_var.get()
        current_function = self.rosenbrock.functions[function_name]
        
        # Получаем все шаги y_j
        points = np.array(self.rosenbrock.all_steps_history)
        
        # Расширение диапазона для отображения
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        
        margin_x = max(abs(x_max - x_min) * 0.3, 1.0) if x_max != x_min else 2.0
        margin_y = max(abs(y_max - y_min) * 0.3, 1.0) if y_max != y_min else 2.0
        
        x_range = (x_min - margin_x, x_max + margin_x)
        y_range = (y_min - margin_y, y_max + margin_y)
        
        # Строим сетку для контурного графика
        x = np.linspace(x_range[0], x_range[1], 200)
        y = np.linspace(y_range[0], y_range[1], 200)
        X, Y = np.meshgrid(x, y)
        
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = current_function([X[i, j], Y[i, j]])
        
        
        levels = self.get_adaptive_levels(Z, num_levels=15) # Автоматическое определение уровней   
        self.ax.contour(X, Y, Z, levels=levels, colors='black', linewidths=0.8, alpha=0.7) # Рисуем контуры    
        # Добавляем подписи для некоторых уровней
        if len(levels) > 0:
            label_levels = levels[::3]
            if label_levels:
                self.ax.clabel(self.ax.contour(X, Y, Z, levels=label_levels, colors='black', linewidths=0.8, alpha=0.7), inline=True, fontsize=8, colors='black', fmt='%.0f')
        
        # Рисуем траекторию из всех шагов
        self.ax.plot(points[:, 0], points[:, 1], 'k-o', linewidth=2, markersize=6, markerfacecolor='black', markeredgecolor='black')
        
        # Добавляем номера для точек X_k
        for i, (k, xk, f_xk, x_next, f_next) in enumerate(self.rosenbrock.points_history):
            x, y = xk
            self.ax.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points', fontsize=10, color='black', fontweight='bold')
        
        # Добавляем оптимальную точку (последняя точка в all_steps_history)
        if len(self.rosenbrock.all_steps_history) > 0:
            opt_point = self.rosenbrock.all_steps_history[-1]
            self.ax.plot(opt_point[0], opt_point[1], 'ro', markersize=8, markerfacecolor='blue', markeredgecolor='black', linewidth=2)

        # Настраиваем внешний вид графика
        self.ax.set_xlabel('x₁', fontsize=10, color='black')
        self.ax.set_ylabel('x₂', fontsize=10, color='black')
        self.ax.grid(True, alpha=0.3, color='gray')
        self.ax.set_facecolor('white')
        
        self.ax.spines['bottom'].set_color('black')
        self.ax.spines['top'].set_color('black')
        self.ax.spines['left'].set_color('black')
        self.ax.spines['right'].set_color('black')
        self.ax.tick_params(colors='black')
        
        self.ax.set_aspect('auto')
        self.fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.12)
        
        self.canvas.draw()
        
    def save_to_csv(self):
        """Сохранение таблицы результатов в CSV файл"""
        try:
            # Проверяем, есть ли данные в таблице
            if not self.tree.get_children():
                messagebox.showwarning("Предупреждение", "Нет данных для сохранения. Сначала запустите алгоритм.")
                return
            
            # Открываем диалог выбора файла
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Сохранить таблицу как"
            )
            
            if not filename:
                return
            
            # Получаем заголовки столбцов
            columns = self.tree['columns']
            
            # Открываем файл для записи
            with open(filename, 'w', newline='', encoding='utf-8-sig') as file:
                writer = csv.writer(file, delimiter=';')
                
                # Записываем заголовки
                writer.writerow(columns)
                
                # Записываем данные
                for item in self.tree.get_children():
                    values = self.tree.item(item)['values']
                    writer.writerow(values)
            
            messagebox.showinfo("Успех", f"Таблица успешно сохранена в файл:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при сохранении файла:\n{str(e)}")
            
    def run_algorithm(self):
        try:
            # Получаем параметры
            eps = float(self.eps_var.get().replace(',', '.'))
            
            x_start = [
                float(self.x1_1.get().replace(',', '.')),
                float(self.x1_2.get().replace(',', '.'))
            ]
            
            directions = [
                [float(self.d1_1.get().replace(',', '.')), float(self.d1_2.get().replace(',', '.'))],
                [float(self.d2_1.get().replace(',', '.')), float(self.d2_2.get().replace(',', '.'))]
            ]
            
            # Получаем интервал для поиска λ
            a = float(self.interval_a.get().replace(',', '.'))
            b = float(self.interval_b.get().replace(',', '.'))
            search_interval = (a, b)
            
            # Проверяем корректность интервала
            if a >= b:
                messagebox.showerror("Ошибка", "Левая граница интервала (a) должна быть меньше правой (b)")
                return
            
            # Проверяем ортогональность направлений
            dot = directions[0][0] * directions[1][0] + directions[0][1] * directions[1][1]
            if abs(dot) > 1e-6:
                messagebox.showwarning("Предупреждение", "Направления должны быть ортогональны!\nПродолжаем выполнение, но результат может быть некорректным.")
            
            # Выбор функции
            function_name = self.function_var.get()
            self.rosenbrock.current_function = self.rosenbrock.functions[function_name]
            
            # Запуск алгоритма
            table_data = self.rosenbrock.run_algorithm(x_start, directions, eps, search_interval)
                   
            for item in self.tree.get_children(): # Обновление таблицы
                self.tree.delete(item)
            
            for row in table_data:
                values = [row[col] for col in self.tree['columns']]
                self.tree.insert('', tk.END, values=values)
              
            self.update_chart() # Обновляем график         
            self.update_result_display() # Обновляем отображение результата

        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка:\n{str(e)}")

def main():
    root = tk.Tk()
    app = RosenbrockGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()