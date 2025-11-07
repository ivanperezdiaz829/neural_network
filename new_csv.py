import csv

# --- Configuración ---
# Cambia estos nombres de archivo según tus necesidades
archivo_entrada = "yolo_labels_dataset.csv"  # El archivo CSV original
archivo_salida = "datos_corregidos.csv"  # El nuevo archivo que se creará
# ---------------------

filas_eliminadas = 0
filas_modificadas = 0
filas_totales_escritas = 0

print(f"Procesando '{archivo_entrada}'...")

try:
    with open(archivo_entrada, mode='r', newline='') as infile, \
         open(archivo_salida, mode='w', newline='') as outfile:

        # Crear lector y escritor de CSV
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # 1. Leer y escribir la cabecera (header)
        try:
            header = next(reader)
            writer.writerow(header)
        except StopIteration:
            print("Error: El archivo de entrada está vacío.")
            exit()

        # 2. Procesar cada fila de datos
        for row in reader:
            if not row:  # Omitir filas vacías
                continue

            try:
                # Obtener el índice de clase (columna 1, es decir, la segunda)
                # El índice de clase es un string, hay que convertirlo a int
                clase_index = int(row[1])

                # --- Aplicar la lógica de re-mapeo ---
                
                if clase_index == 2:
                    # Regla 1: Eliminar la clase 2
                    filas_eliminadas += 1
                    continue  # 'continue' salta al siguiente ciclo (no escribe la fila)

                elif clase_index > 2:
                    # Regla 2: Restar 1 a las clases > 2
                    clase_index -= 1
                    filas_modificadas += 1
                    # Actualizar la fila con el nuevo valor (convertido a string)
                    row[1] = str(clase_index)
                
                # else: (Si la clase es 0 o 1, no se hace nada y se escribe tal cual)
                
                # Escribir la fila (modificada o no) en el archivo de salida
                writer.writerow(row)
                filas_totales_escritas += 1

            except ValueError:
                print(f"Advertencia: Se omitió una fila mal formada (índice no numérico): {row}")
            except IndexError:
                print(f"Advertencia: Se omitió una fila mal formada (faltan columnas): {row}")

    print("\n--- ¡Proceso completado! ---")
    print(f"Resultados guardados en: '{archivo_salida}'")
    print(f"Filas eliminadas (clase 2): {filas_eliminadas}")
    print(f"Filas re-mapeadas (> 2):  {filas_modificadas}")
    print(f"Total de filas escritas (sin cabecera): {filas_totales_escritas}")

except FileNotFoundError:
    print(f"Error: No se encontró el archivo de entrada '{archivo_entrada}'.")
except Exception as e:
    print(f"Ocurrió un error inesperado: {e}")