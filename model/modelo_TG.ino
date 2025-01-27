#include <SD.h>
#include <SPI.h>
#include "AudioTools.h"
#include "mejor_modelo.h"
#include "esp_heap_caps.h"
#include "TensorFlowLite_ESP32.h"
#include "AudioTools/AudioLibs/AudioSourceSD.h"
#include "AudioTools/AudioCodecs/CodecMP3Helix.h"
#include "AudioTools/AudioLibs/AudioBoardStream.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

const int CHANNELS = 1;
const int RES_SIZE = 32;
const int FRAME_STEP = 128;
const int FRAME_LENGTH = 256;
const int SAMPLE_RATE = 16000;
const int TARGET_LENGTH = 8000;
const int BITS_PER_SAMPLE = 16;
const char* FILE_NAME = "/rec.wav";
const int NUM_COLS = FRAME_LENGTH / 2 + 1;
const float ANGLE_BASE = -2.0f * M_PI / FRAME_LENGTH;
const char* labels[] = {"abre", "apaga", "cierra", "dime", "enciende"};
const char* const FILE_PATHS[] = {"/abre/", "/apaga/", "/cierra/", "/dime/", "/enciende/"};

File audioFile;
MP3DecoderHelix decoder;
AudioBoardStream audioKit(AudioKitEs8388V1);
AudioSourceSD source("/", "mp3", PIN_AUDIO_KIT_SD_CARD_CS);
AudioInfo audioInfo(SAMPLE_RATE, CHANNELS, BITS_PER_SAMPLE);
EncodedAudioStream audioOutput(&audioFile, new WAVEncoder());
StreamCopy audioCopier(audioOutput, audioKit);
AudioPlayer player(source, audioKit, decoder);

constexpr size_t kTensorArenaSize = 30 * 1024;
uint8_t *tensor_arena;
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
tflite::AllOpsResolver resolver;
tflite::MicroInterpreter* interpreter = nullptr;

bool isRecording = false;
unsigned long recordingStartTime = 0;
const unsigned long RECORDING_DURATION = 3000;

float* spectrogram = nullptr;
float* spectrogramReSize = nullptr;

void initializePSRAM() {
  if (psramFound()) {
    Serial.println("PSRAM encontrada e inicializada correctamente.");
  } else {
    Serial.println("Error: PSRAM no encontrada. Verifica la configuración del hardware.");
    while (true);
  }
}

void allocateSpectrogramMemory(int numFrames) {
  // Libera memoria previa si existía
  if (spectrogram) {
    free(spectrogram);
  }
  if (spectrogramReSize) {
    free(spectrogramReSize);
  }

  // Asignar memoria para spectrogram y spectrogramReSize en SPIRAM
  spectrogram = (float*)heap_caps_malloc(numFrames * NUM_COLS * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  spectrogramReSize = (float*)heap_caps_malloc(RES_SIZE * RES_SIZE * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

  if (!spectrogram || !spectrogramReSize) {
    Serial.println("Error: No se pudo asignar memoria en SPIRAM para los espectrogramas.");
    while (true); // Detener si no hay memoria suficiente
  } else {
    Serial.printf("Memoria asignada para espectrogram: %d bytes\n", numFrames * NUM_COLS * sizeof(float));
    Serial.printf("Memoria asignada para espectrogramReSize: %d bytes\n", RES_SIZE * RES_SIZE * sizeof(float));
  }
}

void manageSD(const char* filename, const char* mode) {
  Serial.println("Inicializando la tarjeta SD...");

  if (!SD.begin(PIN_AUDIO_KIT_SD_CARD_CS)) {
    Serial.println("Error: No se pudo inicializar la tarjeta SD.");
    return;
  }

  if (mode == FILE_WRITE && SD.exists(filename)) {
    SD.remove(filename);
  }

  audioFile = SD.open(filename, mode);
  if (!audioFile) {
    Serial.println("Error: No se pudo abrir el archivo.");
  }
}

void handleButton3(bool, int, void*) {
  if (isRecording) {
    stopRecording();
    analyzeRecordedAudio();
  } else {
    startRecording();
  }
}

void startRecording() {
  manageSD(FILE_NAME, FILE_WRITE);
  audioOutput.begin(audioInfo);
  isRecording = true;
  recordingStartTime = millis(); 
  Serial.println("Grabación iniciada...");
}

void stopRecording() {
  if (audioFile) {
    audioFile.flush();
    audioFile.close();
    Serial.println("Grabación detenida.");
  }
  isRecording = false;
}

void handleButton4(bool, int, void*) {
  if (!isRecording) {
    analyzeRecordedAudio();
  } else {
    Serial.println("Error: Detenga la grabación antes de analizar el audio.");
  }
}

void playAudio(int indexPath) {
    player.stop();
    source.setPath(FILE_PATHS[indexPath]);
    player.begin();
    player.setAutoNext(false);
}

void dft(const float input[], float* output) {
  for (int k = 0; k < FRAME_LENGTH / 2; ++k) { // Solo hasta la mitad más uno (N/2 + 1)
    float realPart = 0.0f;
    float imagPart = 0.0f;

    for (int n = 0; n < FRAME_LENGTH; ++n) {
      float angle = ANGLE_BASE * n * k;
      realPart += input[n] * cos(angle);
      imagPart += input[n] * sin(angle);
    }

    output[k] = sqrt(realPart * realPart + imagPart * imagPart);
  }
}

void exportSpectrogramToCSV(float* spectrogram, int numFrames, const char* filename) {
  Serial.println("Exportando espectrograma a CSV...");
  manageSD(filename, FILE_WRITE);

  for (int i = 0; i < numFrames; ++i) {
    for (int j = 0; j < NUM_COLS; ++j) {
      int index = i * NUM_COLS + j;
      audioFile.print(spectrogram[index], 8);
      if (j < NUM_COLS - 1) {
        audioFile.print(",");
      }
    }
    audioFile.println();
  }

  audioFile.close();
  Serial.println("Exportación completa");
}

void normalizeSpectrogram(float* spectrogram, int numFrames) {
  Serial.println("Normalizando espectrograma...");

  // Encontrar el valor mínimo y máximo en el espectrograma
  float min_val = spectrogram[0];
  float max_val = -spectrogram[0];
  
  for (int i = 0; i < numFrames * NUM_COLS; ++i) {
    if (spectrogram[i] < min_val) {
      min_val = spectrogram[i];
    }
    if (spectrogram[i] > max_val) {
      max_val = spectrogram[i];
    }
  }
  
  Serial.print("Minimo: ");
  Serial.println(min_val, 6);
  Serial.print("Maximo: ");
  Serial.println(max_val, 6);

  // Normalizar los valores entre 0 y 255
  for (int i = 0; i < numFrames * NUM_COLS; ++i) {
    spectrogram[i] = (spectrogram[i] - min_val) / (max_val - min_val) * 255.0f;
    spectrogram[i] = static_cast<uint8_t>(spectrogram[i]);
  }

  Serial.println("Normalización completada.");
}

void bilinearResize(float* inputMatrix, int inWidth, float* outputMatrix) {
  Serial.println("Redimensionando espectrograma...");
  for (int y = 0; y < RES_SIZE; y++) {
    for (int x = 0; x < RES_SIZE; x++) {
      float gx = ((float)x / (float)(RES_SIZE - 1)) * (NUM_COLS - 1); // (inWidth - 1);
      float gy = ((float)y / (float)(RES_SIZE - 1)) * (inWidth - 1); // (NUM_COLS - 1);
      int x1 = floor(gx);
      int y1 = floor(gy);
      int x2 = min(x1 + 1, NUM_COLS - 1); // inWidth - 1);
      int y2 = min(y1 + 1, inWidth - 1); // NUM_COLS - 1);
      float dx = gx - x1;
      float dy = gy - y1;
      float q11 = inputMatrix[y1 * NUM_COLS + x1]; // inWidth + x1];
      float q12 = inputMatrix[y2 * NUM_COLS + x1];
      float q21 = inputMatrix[y1 * NUM_COLS + x2];
      float q22 = inputMatrix[y2 * NUM_COLS + x2];
      float top = (1 - dx) * q11 + dx * q21;
      float bottom = (1 - dx) * q12 + dx * q22;
      float value = (1 - dy) * top + dy * bottom;
      value = max(0.0f, min(value, 255.0f));
      outputMatrix[y * RES_SIZE + x] = static_cast<uint8_t>(value);
    }
  }
}

void computeHannWindow(float* window) {
  Serial.println("Calculando ventana de Hann...");
  for (int j = 0; j < FRAME_LENGTH; j++) {
    window[j] = 0.5f * (1.0f - cos(2.0f * M_PI * j / (FRAME_LENGTH - 1)));
  }
}

void processFrames(const std::vector<float>& waveform, float* spectrogram, const float* window, int numFrames) {
  float dftOutput[2 * FRAME_LENGTH];
  Serial.println("Creando espectrograma...");

  for (int i = 0; i < numFrames; i++) {
    if (i % 10 == 0) Serial.printf("Procesando frame %d/%d...\n", i + 1, numFrames);

    int startIdx = i * FRAME_STEP;

    float windowedFrame[FRAME_LENGTH];
    for (int j = 0; j < FRAME_LENGTH; j++) {
      windowedFrame[j] = waveform[startIdx + j] * window[j];
    }

    dft(windowedFrame, dftOutput);

    for (int j = 0; j < NUM_COLS; ++j) {
      spectrogram[i * NUM_COLS + j] = dftOutput[j];
    }
  }
}

void setupTensorInput(TfLiteTensor* input, const float* spectrogram, int size) {
  Serial.println("Configurando tensor de entrada...");
  
  Serial.printf("Número de dimensiones: %d\n", input->dims->size);
  Serial.print("Dimensiones de entrada: ");
  for (int i = 0; i < input->dims->size; i++) {
    Serial.print(input->dims->data[i]);
    if (i < input->dims->size - 1) {
      Serial.print("x");
    }
  }
  Serial.println();
  
  Serial.printf("Tipo de dato: %d\n", input->type);

  for (int i = 0; i < size; i++) {
    int8_t quantized_value = (int8_t)(spectrogram[i] - 128);
    input->data.int8[i] = quantized_value;
  }
}

bool runInference() {
  Serial.println("Ejecutando inferencia...");
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("¡Error al ejecutar la inferencia!");
    return false;
  }
  Serial.println("Inferencia ejecutada correctamente.");
  return true;
}

int processInferenceResults(TfLiteTensor* output) {
  Serial.println("Detalles del tensor de salida:");
  Serial.printf("Número de dimensiones: %d\n", output->dims->size);
  Serial.print("Dimensiones de salida: ");
  for (int i = 0; i < output->dims->size; i++) {
    Serial.print(output->dims->data[i]);
    if (i < output->dims->size - 1) {
      Serial.print("x");
    }
  }
  Serial.println();

  if (output->data.raw == nullptr) {
    Serial.println("Error: Puntero a los datos es nulo.");
    return -1;
  }

  int predictedIndex = -1;
  float maxScore = -FLT_MAX;

  if (output->type == kTfLiteInt8) {
    Serial.println("Procesando tensor cuantizado (int8)...");
    Serial.println("Valores del tensor de salida:");
    for (int i = 0; i < output->dims->data[1]; i++) {
      // Convertir de int8 a float usando los parámetros de escala y punto cero
      float score = (output->data.int8[i] - output->params.zero_point) * output->params.scale;

      // Imprimir los valores desescalados
      Serial.print("Clase ");
      Serial.print(i);
      Serial.print(" (");
      Serial.print(labels[i]); // labels debe contener los nombres de las clases
      Serial.print("): ");
      Serial.println(score);

      // Encontrar el índice de la clase con el mayor puntaje
      if (score > maxScore) {
        maxScore = score;
        predictedIndex = i;
      }
    }
  } else {
    Serial.println("Error: Tipo de tensor no compatible.");
    return -1;
  }

  return predictedIndex;
}


void analyzeRecordedAudio() {
  Serial.println("Iniciando análisis de comando...");
  
  float window[FRAME_LENGTH];
  std::vector<float> audioData;

  Serial.println("Preparando la señal...");

  manageSD(FILE_NAME, FILE_READ);

  audioFile.seek(44);

  Serial.print("Tamaño del archivo grabado: ");
  Serial.println(audioFile.size());

  while (audioFile.available()) {
    int16_t sample;
    audioFile.read(reinterpret_cast<uint8_t*>(&sample), sizeof(int16_t));
    audioData.push_back(sample / 32767.0f);  // Dividir por 32768 para valores entre -1 y 1
  }

  audioFile.close();

  if (audioData.size() < TARGET_LENGTH) {
    audioData.resize(TARGET_LENGTH, 0.0f);
  }

  Serial.print("Tamaño del vector audioData: ");
  Serial.println(audioData.size());

  int numFrames = (audioData.size() - FRAME_LENGTH) / FRAME_STEP + 1;

  Serial.printf("Número de frames: %d\n", numFrames);
  Serial.printf("Columnas: %d\n", NUM_COLS);

  allocateSpectrogramMemory(numFrames);

  Serial.println("Generando espectrograma...");
  computeHannWindow(window);
  processFrames(audioData, spectrogram, window, numFrames);
  exportSpectrogramToCSV(spectrogram, numFrames, "/spectrogram.csv");

  normalizeSpectrogram(spectrogram, numFrames);

  bilinearResize(spectrogram, numFrames, spectrogramReSize);

  Serial.println("Espectrograma generado con éxito.");

  TfLiteTensor* input = interpreter->input(0);

  Serial.printf("Tipo de datos del tensor de entrada: %d\n", input->type);

  setupTensorInput(input, spectrogramReSize, RES_SIZE * RES_SIZE);

  if (!runInference()) return;

  TfLiteTensor* output = interpreter->output(0);

  Serial.printf("Tipo de datos del tensor de salida: %d\n", output->type);

  int predictedIndex = processInferenceResults(output);

  // Validar y reproducir predicción
  if (predictedIndex >= 0 && predictedIndex < (sizeof(labels) / sizeof(labels[0]))) {
    Serial.println(labels[predictedIndex]);
    Serial.print("Predicción: ");
    playAudio(predictedIndex);
  } else {
    Serial.println("Predicción fuera de rango.");
  }

  Serial.println("Análisis completado.");
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  initializePSRAM();

  Serial.printf("Heap interno disponible: %d bytes\n", ESP.getFreeHeap());
  Serial.printf("PSRAM disponible: %d bytes\n", ESP.getFreePsram());

  auto audioConfig = audioKit.defaultConfig(RXTX_MODE);
  audioConfig.bits_per_sample = BITS_PER_SAMPLE;
  audioConfig.channels = CHANNELS;
  audioConfig.sample_rate = SAMPLE_RATE;
  audioConfig.sd_active = true;
  audioConfig.input_device = ADC_INPUT_LINE2; // Micrófono
  audioKit.begin(audioConfig);

  audioKit.addAction(audioKit.getKey(3), handleButton3);
  audioKit.addAction(audioKit.getKey(4), handleButton4);

  Serial.println("Inicializando TensorFlow Lite para Microcontroladores...");

  tensor_arena = (uint8_t *)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!tensor_arena) {
    Serial.println("Error: No se pudo asignar memoria para el Tensor Arena en PSRAM.");
    while (true); // Detiene el programa si no hay memoria suficiente
  } else {
    Serial.printf("Memoria asignada para Tensor Arena en PSRAM: %d bytes\n", kTensorArenaSize);
  }

  const tflite::Model* model = tflite::GetModel(mejor_modelo_tflite);
  if (model == nullptr) {
    Serial.println("¡Error! No se pudo cargar el modelo.");
    while (1);
  }

  Serial.println("Modelo cargado correctamente.");

  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Error al asignar tensores.");
    while (true);
  } else {
      Serial.println("Modelo cargado y tensores asignados correctamente.");
  }

  Serial.println("Sistema listo. Pulse 3 para grabar o 4 para analizar audio.");
}

void loop() {
  audioKit.processActions();
  player.copy();

  if (isRecording) {
    audioCopier.copy();

    if (millis() - recordingStartTime >= RECORDING_DURATION) {
      stopRecording();
      analyzeRecordedAudio();
    }
  }
}