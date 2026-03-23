import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.optimize import minimize_scalar
import matplotlib.colors as colors

class RosenbrockMethod:
    def __init__(self):
        self.current_function = None
        self.points_history = []
        self.functions = {
            "F₁(x) = (3x₁² + x₂)² + (2x₁ - 3x₂)²": self.function1,
            "F₂(x) = 9x₁² + 16x₂² - 90x₁ - 128x₂": self.function2,
            "F₃(x) = (x₁ - 2)⁴ + (x₁ - 2x₂)²": self.function3
        }
        
    def function1(self, x):
        """Первая тестовая функция"""
        return (3 * x[0]**2 + x[1])**2 + (2 * x[0] - 3 * x[1])**2
    
    def function2(self, x):
        """Вторая тестовая функция"""
        return 9 * x[0]**2 + 16 * x[1]**2 - 90 * x[0] - 128 * x[1]
    
    def function3(self, x):
        """Третья тестовая функция"""
        return (x[0] - 2)**4 + (x[0] - 2 * x[1])**2
    
    def minimize_along_direction(self, x, direction):
        """Одномерная минимизация вдоль заданного направления"""
        def objective(alpha):
            return self.current_function(x + alpha * direction)
        
        result = minimize_scalar(objective, method='brent')
        return result.x
    
    def run_algorithm(self, x_start, directions, eps):
        """Запуск алгоритма Розенброка"""
        self.points_history.clear()
        
        # Сохраняем начальную точку
        xk = np.array(x_start, dtype=float)
        d = np.array(directions, dtype=float)
        
        # Таблица данных
        table_data = []
        
        k = 1
        first_iteration = True
        while True:
            y = xk.copy()
            lambda_values = []
            
            # Поиск по направлениям
            for j in range(2):
                dj = d[j]
                lambda_j = self.minimize_along_direction(y, dj)
                lambda_values.append(lambda_j)
                
                y_next = y + lambda_j * dj
                
                if j == 0:
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
                else:
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
                
                y = y_next.copy()
            
            x_next = y.copy()
            self.points_history.append((k, xk.copy(), self.current_function(xk), x_next.copy(), self.current_function(x_next)))
            
            # Проверка останова
            delta_x = np.linalg.norm(x_next - xk)
            if delta_x <= eps:
                break
            
            # Перестройка направлений
            a1 = np.zeros(2)
            a2 = np.zeros(2)
            
            if abs(lambda_values[0]) > 1e-10:
                a1 = lambda_values[0] * d[0] + lambda_values[1] * d[1]
            else:
                a1 = d[0].copy()
            
            if abs(lambda_values[1]) > 1e-10:
                a2 = lambda_values[1] * d[1]
            else:
                a2 = d[1].copy()
            
            # Процесс ортогонализации Грама-Шмидта
            new_d = np.zeros((2, 2))
            
            # Первое направление
            b1 = a1.copy()
            norm_b1 = np.linalg.norm(b1)
            new_d[0] = b1 / norm_b1 if norm_b1 > 0 else b1
            
            # Второе направление
            dot = np.dot(a2, new_d[0])
            b2 = a2 - dot * new_d[0]
            norm_b2 = np.linalg.norm(b2)
            new_d[1] = b2 / norm_b2 if norm_b2 > 0 else b2
            
            d = new_d
            xk = x_next
            k += 1
        
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
        input_frame = ttk.LabelFrame(top_frame, text="Параметры алгоритма", padding="10")
        input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N), padx=(0, 10))
        
        # Выбор функции
        ttk.Label(input_frame, text="Выберите функцию:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.function_var = tk.StringVar()
        function_combo = ttk.Combobox(input_frame, textvariable=self.function_var, 
                                    values=list(self.rosenbrock.functions.keys()), state="readonly", width=27)
        function_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        function_combo.current(0)
        
        ttk.Separator(input_frame, orient='horizontal').grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # Начальная точка X1
        ttk.Label(input_frame, text="Начальная точка X1:", font=('Arial', 10, 'bold')).grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(10, 5))
        
        # Фрейм для координат X1
        x1_frame = ttk.Frame(input_frame)
        x1_frame.grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(x1_frame, text="x₁ =").grid(row=0, column=0, padx=5)
        self.x1_1 = tk.StringVar(value="0.0")
        ttk.Entry(x1_frame, textvariable=self.x1_1, width=12).grid(row=0, column=1, padx=5)
        
        ttk.Label(x1_frame, text="x₂ =").grid(row=0, column=2, padx=5)
        self.x1_2 = tk.StringVar(value="3.0")
        ttk.Entry(x1_frame, textvariable=self.x1_2, width=12).grid(row=0, column=3, padx=5)
        
        # Разделитель
        ttk.Separator(input_frame, orient='horizontal').grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # Направление d1
        ttk.Label(input_frame, text="Направление d₁:", font=('Arial', 10, 'bold')).grid(row=5, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(10, 5))
        
        # Фрейм для координат d1
        d1_frame = ttk.Frame(input_frame)
        d1_frame.grid(row=6, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(d1_frame, text="d₁₁ =").grid(row=0, column=0, padx=5)
        self.d1_1 = tk.StringVar(value="1.0")
        ttk.Entry(d1_frame, textvariable=self.d1_1, width=12).grid(row=0, column=1, padx=5)
        
        ttk.Label(d1_frame, text="d₁₂ =").grid(row=0, column=2, padx=5)
        self.d1_2 = tk.StringVar(value="0.0")
        ttk.Entry(d1_frame, textvariable=self.d1_2, width=12).grid(row=0, column=3, padx=5)
        
        # Разделитель
        ttk.Separator(input_frame, orient='horizontal').grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # Направление d2
        ttk.Label(input_frame, text="Направление d₂:", font=('Arial', 10, 'bold')).grid(row=8, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(10, 5))
        
        # Фрейм для координат d2
        d2_frame = ttk.Frame(input_frame)
        d2_frame.grid(row=9, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(d2_frame, text="d₂₁ =").grid(row=0, column=0, padx=5)
        self.d2_1 = tk.StringVar(value="0.0")
        ttk.Entry(d2_frame, textvariable=self.d2_1, width=12).grid(row=0, column=1, padx=5)
        
        ttk.Label(d2_frame, text="d₂₂ =").grid(row=0, column=2, padx=5)
        self.d2_2 = tk.StringVar(value="1.0")
        ttk.Entry(d2_frame, textvariable=self.d2_2, width=12).grid(row=0, column=3, padx=5)
        
        # Разделитель
        ttk.Separator(input_frame, orient='horizontal').grid(row=10, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # Точность
        ttk.Label(input_frame, text="Точность (ε):", font=('Arial', 10, 'bold')).grid(row=11, column=0, sticky=tk.W, padx=5, pady=5)
        self.eps_var = tk.StringVar(value="0.01")
        ttk.Entry(input_frame, textvariable=self.eps_var, width=12).grid(row=11, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Кнопка запуска
        self.start_button = ttk.Button(input_frame, text="Запустить алгоритм", command=self.run_algorithm, width=25)
        self.start_button.grid(row=12, column=0, columnspan=2, pady=20)
        
        # Правая часть - график
        chart_frame = ttk.LabelFrame(top_frame, text="График оптимизации", padding="10")
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
        self.tree = ttk.Treeview(table_container, columns=columns, show='headings', height=8)
        
        # Настраиваем заголовки
        column_widths = {'K': 50, 'X_k': 100, 'F(X_k)': 80, 'j': 40, 'y_j': 100, 
                        'f(y_j)': 80, 'd_j': 100, 'λj': 60, 'y_j+1': 100, 'f(y_j+1)': 80}
        
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
        
        # Настраиваем веса для растягивания
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)  # Верхняя панель растягивается
        main_frame.rowconfigure(1, weight=0)  # Таблица не растягивается (фиксированная высота)
        top_frame.columnconfigure(0, weight=0)  # Левая панель фиксированная
        top_frame.columnconfigure(1, weight=1)  # Правая панель (график) растягивается
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        table_container.columnconfigure(0, weight=1)
        table_container.rowconfigure(0, weight=1)
    
    def plot_function_contour(self, ax, x_range, y_range, function):
        """Построение контурного графика функции"""
        x = np.linspace(x_range[0], x_range[1], 100)
        y = np.linspace(y_range[0], y_range[1], 100)
        X, Y = np.meshgrid(x, y)
        
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = function([X[i, j], Y[i, j]])
        
        # Рисуем контуры черным цветом
        contour = ax.contour(X, Y, Z, levels=20, colors='black', linewidths=0.5, alpha=0.6)
             
        return X, Y, Z
    
    def update_chart(self):
        """Обновление графика"""
        self.ax.clear()
        
        if not self.rosenbrock.points_history:
            return
        
        # Получаем выбранную функцию
        function_name = self.function_var.get()
        current_function = self.rosenbrock.functions[function_name]
        
        # Определяем диапазон для графика на основе точек
        points = []
        for iteration in self.rosenbrock.points_history:
            k, xk, f_xk, x_next, f_next = iteration
            points.append(xk)
        
        points = np.array(points)
        
        # Расширяем диапазон для отображения
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
        
        # Находим минимальное значение функции в области (приблизительный оптимум)
        z_min = np.min(Z)
        z_max = np.max(Z)
        
        # Определяем уровни для малых значений (вблизи оптимума)
        small_levels = []
        # Добавляем уровни: 0.1, 0.25, 0.5, 1, 2, 3, 5, 10, 15, 20, 30, 40, 50
        target_levels = [0.1, 0.25, 0.5, 1, 2, 3, 5, 10, 15, 20, 30, 40, 50]
        
        # Выбираем только те уровни, которые находятся в диапазоне значений функции
        levels = []
        for level in target_levels:
            if level >= z_min and level <= z_max:
                levels.append(level)
            elif level > z_max:
                break
        
        # Если уровней мало, добавляем промежуточные
        if len(levels) < 5:
            # Добавляем линейные уровни между z_min и z_max
            linear_levels = np.linspace(z_min, min(z_max, 20), 10)
            for level in linear_levels:
                if level not in levels and level >= z_min:
                    levels.append(level)
        
        levels = sorted(set(levels))
        
        # Рисуем контуры черным цветом с выбранными уровнями
        contour = self.ax.contour(X, Y, Z, levels=levels, colors='black', linewidths=0.8, alpha=0.7)
        
        # Добавляем подписи только для некоторых уровней, чтобы не загромождать
        # Подписываем каждый 2-й или 3-й уровень
        if len(levels) > 0:
            label_levels = levels[::2]  # Каждый второй уровень
            # Фильтруем, чтобы не подписывать слишком маленькие и слишком большие
            label_levels = [l for l in label_levels if 0.5 <= l <= 50]
            if label_levels:
                self.ax.clabel(contour, levels=label_levels, inline=True, fontsize=8, colors='black', fmt='%d')
        
        # Рисуем траекторию
        self.ax.plot(points[:, 0], points[:, 1], 'k-o', linewidth=2, markersize=6, markerfacecolor='black', markeredgecolor='black')
        
        # Добавляем номера итераций
        for i, (x, y) in enumerate(points):
            self.ax.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points', 
                        fontsize=9, color='black', fontweight='bold')
        
        # Настраиваем внешний вид графика
        self.ax.set_xlabel('x₁', fontsize=10, color='black')
        self.ax.set_ylabel('x₂', fontsize=10, color='black')
        self.ax.grid(True, alpha=0.3, color='gray')
        self.ax.set_facecolor('white')
        
        # Настраиваем цвета осей
        self.ax.spines['bottom'].set_color('black')
        self.ax.spines['top'].set_color('black')
        self.ax.spines['left'].set_color('black')
        self.ax.spines['right'].set_color('black')
        self.ax.tick_params(colors='black')
        
        # Устанавливаем соотношение сторон
        self.ax.set_aspect('auto')
        
        # Уменьшаем отступы вокруг графика
        self.fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.12)
        
        self.canvas.draw()
        
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
            
            # Проверяем ортогональность направлений
            dot = directions[0][0] * directions[1][0] + directions[0][1] * directions[1][1]
            if abs(dot) > 1e-6:
                messagebox.showwarning("Предупреждение", 
                                      "Направления должны быть ортогональны!\nПродолжаем выполнение, но результат может быть некорректным.")
            
            # Выбираем функцию
            function_name = self.function_var.get()
            self.rosenbrock.current_function = self.rosenbrock.functions[function_name]
            
            # Запускаем алгоритм
            table_data = self.rosenbrock.run_algorithm(x_start, directions, eps)
            
            # Обновляем таблицу
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            for row in table_data:
                values = [row[col] for col in self.tree['columns']]
                self.tree.insert('', tk.END, values=values)
            
            # Обновляем график
            self.update_chart()
            
            messagebox.showinfo("Готово", f"Алгоритм завершен.\nНайдено {len(self.rosenbrock.points_history)} итераций.\n"
                               f"Оптимальная точка: ({self.rosenbrock.points_history[-1][1][0]:.4f}, {self.rosenbrock.points_history[-1][1][1]:.4f})\n"
                               f"Значение функции: {self.rosenbrock.points_history[-1][2]:.4f}")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка:\n{str(e)}")

def main():
    root = tk.Tk()
    app = RosenbrockGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()