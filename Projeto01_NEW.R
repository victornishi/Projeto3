# Formação Cientista de Dados - Novo Projeto com Feedback 1
# Machine Learning em Logística Prevendo o Consumo de Energia de Carros Elétricos
#
# Nome do Aluno: Victor Hugo Nishitani

# Uma empresa da área de transporte e logística deseja migrar sua frota para carros 
# elétricos com o objetivo de reduzir os custos. Antes de tomar a decisão, a empresa 
# gostaria de prever o consumo de energia de carros elétricos com base em diversos 
# fatores de utilização e características dos veículos.
#
# Usando um incrível dataset com dados reais disponíveis publicamente, você deverá 
# construir um modelo de Machine Learning capaz de prever o consumo de energia de 
# carros elétricos com base em diversos fatores, tais como o tipo e número de motores 
# elétricos do veículo, o peso do veículo, a capacidade de carga, entre outros atributos.
#
# Dicionário de Dados:
# 
# Car full name (NomeCarro): Nome completo do carro
# Make (Fabricante): Fabricante
# Model (Modelo): Modelo
# Minimal price (gross) [PLN] (PrecoMinimo): Preço mínimo (bruto) [PLN - Moeda Polonesa]
# Engine power [KM] (PotenciaMotor): Poder/Potência do motor [KM - Medida]
# Maximum torque [Nm] (TorqueMaximo): Torque Máximo [Nm - Newton-metro]
# Type of brakes (TipoFreios): Tipo de freios
# Drive type (TipoDirecao): Tipo de Direção
# Battery capacity [kWh] (CapacidadeBateria): Capacidade da bateria [kWh - Quilowatt-hora]
# Range (WLTP) [km] (ConsumoEnergia): Alcance/Autonomia/Consumo de Energia [km - Quilômetros]
# Wheelbase [cm] (DistanciaEixos): Distância entre os eixos [cm - Centímetros]
# Length [cm] (Comprimento): Comprimento [cm - Centímetros]
# Width [cm] (Largura): Largura [cm - Centímetros]
# Height [cm] (Altura): Altura [cm - Centímetros]
# Minimal empty weight [kg] (PesoMinimoVazio): Peso mínimo vazio [kg - Quilogramas]
# Permissable gross weight [kg] (PesoPermitido): Peso bruto permitido [kg - Quilogramas]
# Maximum load capacity [kg] (CapacidadeMaximaCarga): Capacidade máxima de carga [kg - Quilogramas]
# Number of seats (NumeroAssentos): Número de assentos
# Number of doors (NumeroPortas): Número de portas
# Tire size [in] (TamanhoPneu): Tamanho do pneu [in - Polegadas]
# Maximum speed [kph] (VelocidadeMaxima): Velocidade máxima [kph - Quilômetros por hora]
# Boot capacity (VDA) [l] (VolumePortaMalas): Volume do porta-malas (VDA - Medição) [l - Litros]
# Acceleration 0-100 kph [s] (Aceleracao_0_100): Aceleração de 0 a 100 kph [s - Segundos]
# Maximum DC charging power [kW] (PotenciaMaximaCarregamentoDC): Potência Máxima de Carregamento DC 
#                                                                (Corrente Contínua) [kW - Quilowatts]
# mean - Energy consumption [kWh/100 km] (MediaConsumoEnergia): Média - Consumo de energia [kWh/100 km]

# Configurando o diretório de trabalho
setwd("/Users/nishi/Desktop/FCD/BigDataRAzure/Cap20/Projeto01")
getwd()

# Carrega os pacotes na sessão R
library(zoo)
library(data.table)
library(lattice)
library(psych)
library(randomForest)
library(ggplot2)
library(caTools)
library(MLmetrics)
library(neuralnet)


## Etapa 1 - Coletando os Dados
##### Carga dos Dados ##### 

# Carregamos o dataset antes da transformação
dados <- read.csv("dataset_eletric_cars.csv", stringsAsFactors = F, 
               sep = ";", dec = ",", header = T)


## Etapa 2 - Pré-Processamento
##### Análise Exploratória dos Dados - Limpeza e Organização de Dados ##### 

# Visualizamos os dados
View(dados)
dim(dados)
str(dados)

# Renomeamos as colunas com nomes mais amigáveis
myColumns <- colnames(dados)
myColumns

myColumns[1] <- "NomeCarro"
myColumns[2] <- "Fabricante"
myColumns[3] <- "Modelo"
myColumns[4] <- "PrecoMinimo"
myColumns[5] <- "PotenciaMotor"
myColumns[6] <- "TorqueMaximo"
myColumns[7] <- "TipoFreios"
myColumns[8] <- "TipoDirecao"
myColumns[9] <- "CapacidadeBateria"
myColumns[10] <- "Autonomia"
myColumns[11] <- "DistanciaEixos"
myColumns[12] <- "Comprimento"
myColumns[13] <- "Largura"
myColumns[14] <- "Altura"
myColumns[15] <- "PesoMinimoVazio"
myColumns[16] <- "PesoPermitido"
myColumns[17] <- "CapacidadeMaximaCarga"
myColumns[18] <- "NumeroAssentos"
myColumns[19] <- "NumeroPortas"
myColumns[20] <- "TamanhoPneu"
myColumns[21] <- "VelocidadeMaxima"
myColumns[22] <- "VolumePortaMalas"
myColumns[23] <- "Aceleracao_0_100"
myColumns[24] <- "PotenciaMaximaCarregamentoDC"
myColumns[25] <- "MediaConsumoEnergia"

myColumns

colnames(dados) <- myColumns
rm(myColumns)

# Verificando os valores únicos nas variáveis
length(unique(dados$NomeCarro))
length(unique(dados$Fabricante))
length(unique(dados$Modelo))

length(unique(dados$TipoFreios))
length(unique(dados$TipoDirecao))
table(dados$TipoFreios)
table(dados$TipoDirecao)

# Como a variável NomeCarro contém os dados das variáveis Fabricante e Modelo,
# optamos por removê-las para não ter dados repetidos no nosso dataset
dados$Fabricante <- NULL
dados$Modelo <- NULL
dim(dados)

# Na variável "TipoFreios", vamos transformar de categórica para sua representação
# numérica usando label enconding, Também, aproveitaremos para tratar os valores
# "Vazios" com o valor de maior frequência 'disc (front + rear)', ou seja, '0'
dados1 <- dados
dados1["TipoFreios"][dados1["TipoFreios"] == 'disc (front + rear)'] <- 1
dados1["TipoFreios"][dados1["TipoFreios"] == 'disc (front) + drum (rear)'] <- 2
dados1["TipoFreios"][dados1["TipoFreios"] == ''] <- 1
dados1$TipoFreios <- as.integer(factor(dados1$TipoFreios))

# Na variável "TipoDirecao", vamos transformar de categórica para sua representação
# numérica usando label enconding
dados1$TipoDirecao <- as.integer(factor(dados1$TipoDirecao))

# Verificamos se temos valores ausentes
sum(is.na(dados1))
sum(!complete.cases(dados1))
prop.table(table(is.na(dados1)))

# Função que exibe as colunas com valores ausentes no dataframe
nacols <- function(df) {
  colnames(df)[unlist(lapply(df, function(x) any(is.na(x))))]
}
nacols(dados1)

# Para a variável MediaConsumoEnergia alvo (target) vamos remover os 9 registros ausentes
sum(is.na(dados1$MediaConsumoEnergia))
dim(dados1)
dados1 <- dados1[!is.na(dados1$MediaConsumoEnergia),]
dim(dados1)

# Na variável "NomeCarro", vamos transformar de categórica para sua representação
# numérica usando label enconding
dados1$NomeCarro <- as.integer(factor(dados1$NomeCarro))

# Para as variáveis preditoras com valores ausentes, vamos substituir os dados ausentes 
# pela média dos valores de cada coluna
dados1[] <- lapply(dados1, na.aggregate) # Pacote zoo

str(dados1)
nacols(dados1)
View(dados1)

# Verificamos se temos valores vazios
colSums(is.na(dados1) | dados1 == "")

# Verificamos se temos valores duplicados
sum(duplicated(dados1))


### Análises entre a variável target e algumas variáveis preditoras ###

# Análise 1 - Relação entre "Potência do Motor" e "Média do Consumo de Energia"

# Médias de Tendência Central das variáveis PotenciaMotor e MediaConsumoEnergia
summary(dados1$PotenciaMotor)
summary(dados1$MediaConsumoEnergia)

# Construímos os Histogramas para verificar como está a distribuição dos dados
# das variáveis PotenciaMotor e MediaConsumoEnergia
hist(dados1$PotenciaMotor, main = 'Histograma da Potência do Motor', 
     xlab = 'Potência do Motor')

hist(dados1$MediaConsumoEnergia, main = 'Histograma da Média do Consumo de Energia', 
     xlab = 'Média do Consumo de Energia')

# Observando os histogramas, é possível visualizar que os dados não seguem uma 
# distribuíção normal. Aparentemente, há uma assimetria positiva, pois a 
# concentração é maior nos valores menores, ou seja, o gráfico apresenta uma
# cauda mais longa à direita, apesar de aumentar um pouco a frequência no final 
# da cauda do histograma  da variável "MediaConsumoEnergia"

# Construímos os Boxplots para análisar as correlações e possíveis outliers das 
# variáveis PotenciaMotor e MediaConsumoEnergia
boxplot(dados1$PotenciaMotor, main = "Boxplot da Potência do Motor", 
        ylab = "Potência do Motor")

boxplot(dados1$MediaConsumoEnergia, main = "Boxplot da Média do Consumo de Energia", 
        ylab = "Média do Consumo de Energia")

# Construímos um Scatterplot para analisar se há correlação das variáveis
# PotenciaMotor (Preditora) e MediaConsumoEnergia (Target)
ggplot(dados1, aes(x = PotenciaMotor, y = MediaConsumoEnergia)) + 
  geom_point(shape = 1) +
  geom_smooth(method = lm , color = "red", se = FALSE) + 
  xlab("Potência do Motor") +
  ylab("Média do Consumo de Energia") +
  labs(title = "Scatterplot - Média do Consumo de Energia vs Potência do Motor")

# No gráfico de dispersão, parece haver correlação positiva entre as duas variáveis.
# Podemos confirmar calculando o coeficiente de correlação entre elas.

# Este gráfico fornece mais informações sobre o relacionamento entre as variáveis
pairs.panels(dados1[c("PotenciaMotor", "MediaConsumoEnergia")]) # Pacote psych

# Os dados tem uma forte correlação positiva. Sendo assim, pode-se deduzir que a
# Média do Consumo de Energia tende a aumentar se a Potência do Motor é maior

# Análise 2 - Relação entre "Capacidade da Bateria" e "Média do Consumo de Energia"

# Médias de Tendência Central da variável CapacidadeBateria
summary(dados1$CapacidadeBateria)

# Construímos um Histograma para verificar como está a distribuição dos dados
# da variável CapacidadeBateria
hist(dados1$CapacidadeBateria, main = 'Histograma da Capacidade da Bateria', 
     xlab = 'Capacidade da Bateria')

# Observando o histograma, não é possível observar onde está a maior concentração
# dos dados. Sendo assim, a distribuição dos dados é indefinida.

# Construímos um Boxplot para análisar as correlações e possíveis outliers da 
# variável CapacidadeBateria
boxplot(dados1$CapacidadeBateria, main = "Boxplot da Capacidade da Bateria", 
        ylab = "Capacidade da Bateria")

# Construímos um Scatterplot para analisar se há correlação das variáveis
# CapacidadeBateria (Preditora) e MediaConsumoEnergia (Target)
ggplot(dados1, aes(x = CapacidadeBateria, y = MediaConsumoEnergia)) + 
  geom_point(shape = 1) +
  geom_smooth(method = lm , color = "red", se = FALSE) + 
  xlab("Capacidade da Bateria") +
  ylab("Média do Consumo de Energia") +
  labs(title = "Scatterplot - Média do Consumo de Energia vs Capacidade da Bateria")

# No gráfico de dispersão, parece haver correlação positiva entre as duas variáveis.
# Podemos confirmar calculando o coeficiente de correlação entre elas.

# Este gráfico fornece mais informações sobre o relacionamento entre as variáveis
pairs.panels(dados1[c("CapacidadeBateria", "MediaConsumoEnergia")]) # Pacote psych

# Os dados tem uma correlação positiva moderada. Sendo assim, pode-se deduzir que a
# Média do Consumo de Energia tende a aumentar se a Capacidade da Bateria aumenta

# Análise 3 - Relação entre "Autonomia" e "Média do Consumo de Energia"

# Médias de Tendência Central da variável Autonomia
summary(dados1$Autonomia)

# Construímos um Histograma para verificar como está a distribuição dos dados
# da variável Autonomia
hist(dados1$Autonomia, main = 'Histograma da Autonomia', 
     xlab = 'Autonomia')

# Observando o histograma, é possível visualizar que os dados seguem uma 
# distribuíção normal, ou seja, eles estão normalmente distribuídos

# Construímos um Boxplot para análisar as correlações e possíveis outliers das 
# variável Autonomia
boxplot(dados1$Autonomia, main = "Boxplot da Autonomia", 
        ylab = "Autonomia")

# Construímos um Scatterplot para analisar se há correlação das variáveis
# Autonomia (Preditora) e MediaConsumoEnergia (Target)
ggplot(dados1, aes(x = Autonomia, y = MediaConsumoEnergia)) + 
  geom_point(shape = 1) +
  geom_smooth(method = lm , color = "red", se = FALSE) + 
  xlab("Autonomia") +
  ylab("Média do Consumo de Energia") +
  labs(title = "Scatterplot - Média do Consumo de Energia vs Autonomia")

# No gráfico de dispersão, não parece haver correlação entre as duas variáveis.
# Podemos confirmar calculando o coeficiente de correlação entre elas.

# Este gráfico fornece mais informações sobre o relacionamento entre as variáveis
pairs.panels(dados1[c("Autonomia", "MediaConsumoEnergia")]) # Pacote psych

# Os dados tem uma fraca correlação positiva, não havendo uma correlação clara.

# Análise 4 - Relação entre "Capacidade Máxima de Carga" e "Média do Consumo de Energia"

# Médias de Tendência Central da variável CapacidadeMaximaCarga
summary(dados1$CapacidadeMaximaCarga)

# Construímos um Histograma para verificar como está a distribuição dos dados
# da variável CapacidadeMaximaCarga
hist(dados1$CapacidadeMaximaCarga, main = 'Histograma da Capacidade Máxima de Carga', 
     xlab = 'Capacidade Máxima de Carga')

# Observando o histograma, é possível observar uma maior concentração dos dados no
# no centro do gráfico seguindo uma distribuição normal, apesar que parece haver
# outliers à direita

# Construímos um Boxplot para análisar as correlações e possíveis outliers da 
# variável CapacidadeMaximaCarga
boxplot(dados1$CapacidadeMaximaCarga, main = "Boxplot da Capacidade Máxima de Carga", 
        ylab = "Capacidade Máxima de Carga")

# No Boxplot é possível observar melhor a existência do outlier. Posteriormente,
# teremos que tratar esse valor

# Construímos um Scatterplot para analisar se há correlação das variáveis
# CapacidadeMaximaCarga (Preditora) e MediaConsumoEnergia (Target)
ggplot(dados1, aes(x = CapacidadeMaximaCarga, y = MediaConsumoEnergia)) + 
  geom_point(shape = 1) +
  geom_smooth(method = lm , color = "red", se = FALSE) + 
  xlab("Capacidade Máxima de Carga") +
  ylab("Média do Consumo de Energia") +
  labs(title = "Scatterplot - Média do Consumo de Energia vs Capacidade Máxima de Carga")

# No gráfico de dispersão, parece haver correlação positiva entre as duas variáveis.
# Podemos confirmar calculando o coeficiente de correlação entre elas.

# Este gráfico fornece mais informações sobre o relacionamento entre as variáveis
pairs.panels(dados1[c("CapacidadeMaximaCarga", "MediaConsumoEnergia")]) # Pacote psych

# Os dados tem uma correlação positiva moderada. Sendo assim, pode-se deduzir que a
# Média do Consumo de Energia tende a aumentar se a Capacidade Máxima de Carga é maior.

# Análise 5 - Relação entre "Potência Máxima de Carregamento DC" e "Média do Consumo de Energia"

# Médias de Tendência Central da variável PotenciaMaximaCarregamentoDC
summary(dados1$PotenciaMaximaCarregamentoDC)

# Construímos um Histograma para verificar como está a distribuição dos dados
# da variável PotenciaMaximaCarregamentoDC
hist(dados1$PotenciaMaximaCarregamentoDC, 
     main = 'Histograma da Potência Máxima de Carregamento DC', 
     xlab = 'Potência Máxima de Carregamento DC')

# Observando o histograma, é possível observar uma maior concentração dos dados no
# no centro do gráfico seguindo uma distribuição normal, apesar que parece haver
# um outlier à direita

# Construímos um Boxplot para análisar as correlações e possíveis outliers da 
# variável PotenciaMaximaCarregamentoDC
boxplot(dados1$PotenciaMaximaCarregamentoDC, 
        main = "Boxplot da Potência Máxima de Carregamento DC", 
        ylab = "Potência Máxima de Carregamento DC")

# No Boxplot é possível observar melhor a existência do outlier. Posteriormente,
# teremos que tratar esse valor

# Construímos um Scatterplot para analisar se há correlação das variáveis
# PotenciaMaximaCarregamentoDC (Preditora) e MediaConsumoEnergia (Target)
ggplot(dados1, aes(x = PotenciaMaximaCarregamentoDC, y = MediaConsumoEnergia)) + 
  geom_point(shape = 1) +
  geom_smooth(method = lm , color = "red", se = FALSE) + 
  xlab("Potência Máxima de Carregamento DC") +
  ylab("Média do Consumo de Energia") +
  labs(title = "Scatterplot - Média do Consumo de Energia vs Potência Máxima de Carregamento DC")

# No gráfico de dispersão, parece haver correlação positiva entre as duas variáveis.
# Podemos confirmar calculando o coeficiente de correlação entre elas.

# Este gráfico fornece mais informações sobre o relacionamento entre as variáveis
pairs.panels(dados1[c("PotenciaMaximaCarregamentoDC", "MediaConsumoEnergia")]) # Pacote psych

# Os dados tem uma correlação positiva moderada. Sendo assim, pode-se deduzir que a
# Média do Consumo de Energia tende a aumentar se a Potência Máxima de Carregamento DC 
# é maior.

### Análises entre as variáveis preditoras ###
# Análise 1 - Relação entre "Potência do Motor" e "Capacidade da Bateria"

# Construímos um Scatterplot para analisar se há correlação das variáveis
# PotenciaMotor (Preditora) e CapacidadeBateria (Preditora)
ggplot(dados1, aes(x = PotenciaMotor, y = CapacidadeBateria)) + 
  geom_point(shape = 1) +
  geom_smooth(method = lm , color = "red", se = FALSE) + 
  xlab("Potência do Motor") +
  ylab("Capacidade da Bateria") +
  labs(title = "Scatterplot - Capacidade da Bateria vs Potência do Motor")

# No gráfico de dispersão, parece haver uma correlação positiva entre as duas variáveis.
# Podemos confirmar calculando o coeficiente de correlação entre elas.

# Este gráfico fornece mais informações sobre o relacionamento entre as variáveis
pairs.panels(dados1[c("PotenciaMotor", "CapacidadeBateria")]) # Pacote psych

# Os dados tem uma forte correlação positiva. Portanto, nota-se que há 
# multicolinearidade entre as variáveis preditoras, o que é um problema e
# devemos fazer uma escolha entre deixá-las no conjunto de dados ou remover
# uma delas. Neste caso, vamos mantê-las para observar o comportamento de ambas,
# e posteriormente podemos remover uma delas caso nosso modelo apresente problema
# de performance

# Análise 2 - Relação entre "Potência do Motor" e "Capacidade da Bateria"

# Construímos um Scatterplot para analisar se há correlação das variáveis
# Autonomia (Preditora) e PotenciaMotor (Preditora)
ggplot(dados1, aes(x = PotenciaMotor, y = Autonomia)) + 
  geom_point(shape = 1) +
  geom_smooth(method = lm , color = "red", se = FALSE) + 
  xlab("Potência do Motor") +
  ylab("Autonomia") +
  labs(title = "Scatterplot - Potência do Motor vs Autonomia")

# No gráfico de dispersão, parece haver uma correlação positiva entre as duas variáveis.
# Podemos confirmar calculando o coeficiente de correlação entre elas.

# Este gráfico fornece mais informações sobre o relacionamento entre as variáveis
pairs.panels(dados1[c("PotenciaMotor", "Autonomia")]) # Pacote psych

# Os dados tem uma correlação positiva moderada. Apesar de haver multicolinearidade entre
# as variáveis preditoras, vamos mantê-las e observá-las. Posteriormente podemos remover 
# uma delas caso necessário

# Análise 3 - Relação entre "Capacidade da Bateria" e "Capacidade Máxima de Carga"

# Construímos um Scatterplot para analisar se há correlação das variáveis
# CapacidadeBateria (Preditora) e CapacidadeMaximaCarga (Preditora)
ggplot(dados1, aes(x = CapacidadeBateria, y = CapacidadeMaximaCarga)) + 
  geom_point(shape = 1) +
  geom_smooth(method = lm , color = "red", se = FALSE) + 
  xlab("Capacidade da Bateria") +
  ylab("Capacidade Máxima de Carga") +
  labs(title = "Scatterplot - Capacidade da Bateria vs Capacidade Máxima de Carga")

# No gráfico de dispersão, parece haver uma leve correlação positiva entre as duas variáveis.
# Podemos confirmar calculando o coeficiente de correlação entre elas.

# Este gráfico fornece mais informações sobre o relacionamento entre as variáveis
pairs.panels(dados1[c("CapacidadeBateria", "CapacidadeMaximaCarga")]) # Pacote psych

# Os dados tem uma correlação positiva moderada. Apesar de haver multicolinearidade entre
# as variáveis preditoras, vamos mantê-las e observá-las. Posteriormente podemos remover 
# uma delas caso necessário


### Métodos de "Pearson" e "Spearman" - Análise da Correlação entre todas as variáveis ### 

# Obtendo apenas as colunas numéricas
colunas_numericas <- sapply(dados1, is.numeric)
colunas_numericas

# Converte dataset para data.table e separa os dados em outro dataset somente para
# essa análise
dados2 <- as.data.table(dados1)

# Filtrando as colunas numéricas para correlação
data_cor <- cor(dados2[,..colunas_numericas])
data_cor
head(data_cor)

# Definindo as colunas para a análise de correlação 
cols <- colunas_numericas

# Utilizando um vetor com os métodos de correlação "Pearson" e "Spearman"
metodos <- c("pearson", "spearman")

# Aplicando os métodos de correlação com a função cor()
cors <- lapply(metodos, function(method) 
  (cor(dados2[ , ..cols], method = method)))

head(cors)

# Preparando o plot
plot.cors <- function(x, labs){
  diag(x) <- 0.0 
  plot( levelplot(x, 
                  main = paste("Plot de Correlação usando Método", labs),
                  scales = list(x = list(rot = 90), cex = 1.0)) )
}

# Mapa de Correlação
Map(plot.cors, cors, metodos) # Pacote lattice


### Random Forest - Modelo para identificar os atributos com maior importância ###

# Avalidando a importância de todas as variaveis
modelo <- randomForest(MediaConsumoEnergia ~ ., 
                       data = dados1, 
                       ntree = 100, 
                       nodesize = 10,
                       importance = TRUE)

varImpPlot(modelo)


## Etapa 3: Treinando o modelo e Criando o Modelo Preditivo no R

# Vamos dividir os dados em treino e teste, sendo 70% para dados de treino e 
# 30% para dados de teste
set.seed(123)
split = sample.split(dados1$MediaConsumoEnergia, SplitRatio = 0.70)
dados_treino = subset(dados1, split == TRUE)
dados_teste = subset(dados1, split == FALSE)

# Verificando o número de linhas
nrow(dados_treino)
nrow(dados_teste)


### Treinamento do Modelo 1 com Regressão Linear (Benchmark) ###
set.seed(775)
modelo_v1 <- lm(MediaConsumoEnergia ~.,
                data = dados_treino)

## Etapa 4: Avaliando a Performance do Modelo
# Visualizando os coeficientes
summary(modelo_v1)

# Aqui verificamos os gastos previstos pelo modelo que devem ser iguais aos 
# dados de treino
set.seed(345)
previsao_v1 <- data.frame(Real = dados_treino$MediaConsumoEnergia,
                          Previsto = predict(modelo_v1))
View(previsao_v1)

# Prevendo os gastos com Dados de teste
set.seed(456)
previsao_v2 <- data.frame(Real = dados_teste$MediaConsumoEnergia,
                          Previsto = predict(modelo_v1, newdata = dados_teste))
View(previsao_v2)

# Construímos um Scatterplot para analisar se há correlação entre os valores 
# da Média de Consumo de Energia Real do que foi Previsto
ggplot(previsao_v2, aes(x = Real, y = Previsto)) + 
  geom_point(shape = 1) +
# geom_smooth(method = lm , color = "red", se = FALSE) + 
  stat_smooth() +
  xlab("Valor Real") +
  ylab("Valor Previsto") +
  labs(title = "Scatterplot - Valor Real vs Valor Previsto")

# Apartir do gráfica de dispersão, podemos notar que há uma correlação positiva
# entre os valores que foram previstos e os valores reais nos dados de teste.
# Também podemos ver que a maioria dos pontos estão dentro da área sombreada no
# gráfico e próximos a linha azul, isso significa que temos um bom modelo.

# Interpretando o Modelo

# Métricas
set.seed(12)
summary(dados1$MediaConsumoEnergia)
MAE(previsao_v2$Previsto, previsao_v2$Real) # 2.072618

# O MAE prevê que, em média, as previsões do nosso modelo (de Consumo de Enegia) 
# estão erradas em aproximadamente 2.07 kWh/100 km, o que é um valor pequeno  
# comparado ao valor médio de Consumo de Energia dos Carros Elétricos que é de
# 18.99 kWh/100 km.

# Calculando o erro médio ao Quadrado
## Quão distantes seus valores previstos estão dos valores observados
# MSE
set.seed(1)
mse <- mean((previsao_v2$Real - previsao_v2$Previsto)^2)
print(mse) # 10.32757
MSE(previsao_v2$Previsto,previsao_v2$Real)

# RMSE (Raiz Quadrada)
set.seed(2)
caret::RMSE(previsao_v2$Previsto,previsao_v2$Real) # 3.213653
RMSE(previsao_v2$Previsto,previsao_v2$Real)

# O RMSE prevê que, em média, as previsões do nosso modelo (de Consumo de Energia) 
# estão erradas em aproximadamente 3.21 kWh/100 km, que é um valor pequeno comparado 
# ao valor médio 18.99 kWh/100 km de Consumo de Energia dos Carros Elétricos.

# R-squared (Coeficiente de Determinação)
set.seed(3)
caret::R2(previsao_v2$Previsto, previsao_v2$Real) # 0.7080751
R2_Score(previsao_v2$Previsto, previsao_v2$Real) # 0.3142577

# Calculando R Squared
SSE = sum((previsao_v2$Previsto - previsao_v2$Real)^2)
SST = sum((mean(dados1$MediaConsumoEnergia) - previsao_v2$Real)^2)

# R-Squared
# Ajuda a avaliar o nível de precisão do nosos modelo. Quanto maior, melhor
R2 = 1 - (SSE/SST)
R2 # 0.3240212

# Temos a acurácia de 71% do nosso modelo utilizando a "regressão linear" o que é
# bom para um primeiro modelo, mas vamos ver se conseguimos melhorar utilizando
# outros métodos.


## Etapa 5: Otimizando o Modelo preditivo
### Treinamento do Modelo 2 com Random Forest ###
# Vamos utilizar o modelo de "Random Forest" com os dados de treino
set.seed(007)
modelo_v2 <- randomForest(MediaConsumoEnergia ~ .,
                          data = dados_treino, 
                          ntree = 100, 
                          nodesize = 10)

# Avaliando a Performance do Modelo
# Mais detalhes sobre o modelo
print(modelo_v2)

# Agora fazemos as previsões com o modelo usando dados de teste
set.seed(456)
previsao_v3 <- data.frame(Real = dados_teste$MediaConsumoEnergia,
                          Previsto = predict(modelo_v2, newdata = dados_teste))
View(previsao_v3)

# Construímos um Scatterplot para analisar se há correlação entre os valores 
# da Média de Consumo de Energia Real do que foi Previsto
ggplot(previsao_v3, aes(x = Real, y = Previsto)) + 
  geom_point(shape = 1) +
# geom_smooth(method = lm , color = "red", se = FALSE) + 
  stat_smooth() +
  xlab("Valor Real") +
  ylab("Valor Previsto") +
  labs(title = "Scatterplot - Valor Real vs Valor Previsto")

# Apartir do gráfica de dispersão, podemos notar que há uma correlação positiva
# entre os valores que foram previstos e os valores reais nos dados de teste.
# Também podemos ver que a maioria dos pontos estão dentro da área sombreada no
# gráfico e muito próximos da linha azul, comparado ao modelo anterior. 
# Isso significa que, provavelmente, temos um modelo melhor que o da "Regressão Linear".

# Interpretando o Modelo

# Métricas
set.seed(32)
summary(dados1$MediaConsumoEnergia)
MAE(previsao_v3$Previsto, previsao_v3$Real) # 1.174722

# O MAE prevê que, em média, as previsões do nosso modelo (de Consumo de Enegia) 
# estão erradas em aproximadamente 1.18 kWh/100 km, o que é um valor pequeno  
# comparado ao valor médio de Consumo de Energia dos Carros Elétricos que é de
# 18.99 kWh/100 km.

# Calculando o erro médio ao Quadrado
## Quão distantes seus valores previstos estão dos valores observados
# MSE
set.seed(1)
mse <- mean((previsao_v3$Real - previsao_v3$Previsto)^2)
print(mse) # 1.752155
MSE(previsao_v3$Previsto, previsao_v3$Real)

# RMSE (Raiz Quadrada)
set.seed(2)
caret::RMSE(previsao_v3$Previsto,previsao_v3$Real) # 1.32369
RMSE(previsao_v3$Previsto,previsao_v3$Real)

# O RMSE prevê que, em média, as previsões do nosso modelo (de Consumo de Energia) 
# estão erradas em aproximadamente 1.32 kWh/100 km, que é um valor pequeno comparado
# ao valor médio 18.78 kWh/100 km de Consumo de Energia dos Carros Elétricos.

# R-squared (Coeficiente de Determinação)
set.seed(3)
caret::R2(previsao_v3$Previsto, previsao_v3$Real) # 0.8935294
R2_Score(previsao_v3$Previsto, previsao_v3$Real) # 0.8836583

# Calculando R Squared
SSE = sum((previsao_v3$Previsto - previsao_v3$Real)^2)
SST = sum((mean(dados1$MediaConsumoEnergia) - previsao_v3$Real)^2)

# R-Squared
# Ajuda a avaliar o nível de precisão do nosos modelo. Quanto maior, melhor
R2 = 1 - (SSE/SST)
R2 # 0.8853148

# Temos a acurácia de 89% do modelo 2 utilizando o método "Random Forest",
# ou seja, conseguimos melhorar a performance em relação ao modelo 1.


### Treinamento do Modelo 3 com Redes Neurais ###

### Feature Scaling - Normalização dos Dados ###
# Normalizando os dados
dados_norm <- dados1

maxs <- apply(dados_norm, 2, max)
mins <- apply(dados_norm, 2, min)
maxs
mins
dados_norm <- as.data.frame(scale(dados_norm, center = mins, scale = maxs - mins))
head(dados_norm)

# Vamos dividir os dados em treino e teste, sendo 70% para dados de treino e 
# 30% para dados de teste
set.seed(123)
split = sample.split(dados_norm$MediaConsumoEnergia, SplitRatio = 0.70)
dados_treino_nn = subset(dados_norm, split == TRUE)
dados_teste_nn = subset(dados_norm, split == FALSE)

# Verificando o número de linhas
nrow(dados_treino_nn)
nrow(dados_teste_nn)

# Vamos utilizar o modelo de "Redes Neurais" com os dados de treino
set.seed(234)
nn_ <- neuralnet(MediaConsumoEnergia ~ ., 
                 dados_treino_nn, hidden = c(5, 3), 
                 linear.output = TRUE)

plot(nn_)

# Criando o modelo NN
modelo_nn_v1 <- compute(nn_, dados_teste_nn)
str(modelo_nn_v1)
summary(modelo_nn_v1)

# Convertendo os dados de teste
previsoes_v4 <- modelo_nn_v1$net.result * (max(dados1$MediaConsumoEnergia) - min(dados1$MediaConsumoEnergia)) + min(dados1$MediaConsumoEnergia)
test_convert <- (dados_teste_nn$MediaConsumoEnergia) * (max(dados1$MediaConsumoEnergia) - min(dados1$MediaConsumoEnergia)) + min(dados1$MediaConsumoEnergia)
test_convert

# Calculando o Mean Squared Error
set.seed(71)
MSE.nn <- sum((test_convert - previsoes_v4)^2)/nrow(dados_teste_nn)
MSE.nn # 0.8714947
MSE(previsoes_v4, test_convert)

# O RMSE prevê que, em média, as previsões do nosso modelo (de Consumo de Energia) 
# estão erradas em aproximadamente 0.87 kWh/100 km, que é um valor pequeno comparado
# ao valor médio 18.78 kWh/100 km de Consumo de Energia dos Carros Elétricos.

# RMSE (Raiz Quadrada)
set.seed(7)
caret::RMSE(test_convert, previsoes_v4) # 0.9335388
RMSE(previsoes_v4, test_convert)

# R-squared (Coeficiente de Determinação)
set.seed(7)
caret::R2(previsoes_v4, test_convert) # 0.9532607
R2_Score(previsoes_v4, test_convert) # 0.9421334

# Calculando R Squared
SSE = sum((previsoes_v4 - test_convert)^2)
SST = sum((mean(dados1$MediaConsumoEnergia) - test_convert)^2)

# R-Squared
# Ajuda a avaliar o nível de precisão do nosos modelo. Quanto maior, melhor
R2 = 1 - (SSE/SST)
R2 # 0.9429573

# Temos a acurácia de 94% utilizando o modelo de "Redes Neurais" o que é
# excelente, pois supera os 2 modelos anteriores "Regressão Linear" e
# "Random Forest"

# Obtendo os erros de previsão
error.df <- data.frame(test_convert, previsoes_v4)
head(error.df)

# Métricas
summary(dados1$MediaConsumoEnergia)
MAE(previsoes_v4, test_convert) # 0.729069

# O MAE prevê que, em média, as previsões do nosso modelo (de Consumo de Enegia) 
# estão erradas em aproximadamente 0.73 kWh/100 km, o que é um valor pequeno  
# comparado ao valor médio de Consumo de Energia dos Carros Elétricos que é de
# 18.99 kWh/100 km.

# Plot dos erros
ggplot(error.df, aes(x = test_convert, y = previsoes_v4)) +
  geom_point() + stat_smooth()

## Etapa 4: Avaliando e Interpretando o Modelo
# Criando uma tabela cruzada dos dados previstos x dados atuais
results <- data.frame(actual = dados_teste_nn$MediaConsumoEnergia, prediction = modelo_nn_v1$net.result)
roundedresults<-sapply(results,round,digits=0)
roundedresultsdf=data.frame(roundedresults)
attach(roundedresultsdf)
table(actual,prediction)

# Validação Cruzada
set.seed(8)
lm.fit <- glm(MediaConsumoEnergia~., data=dados_teste_nn)
cv.glm(dados_teste_nn, lm.fit, K = 10)$delta[1] # 4.234154


## Considerações Finais
# Plot de Comparação dos 3 Modelos: Regressão Linear, Random Forest e Redes Neurais
par(mfrow=c(1,2))

plot(previsao_v2$Real, previsao_v2$Previsto,col='red',main='Real vs Previsto LM',pch=18,cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend='LM',pch=18,col='red', bty='n')

plot(previsao_v3$Real, previsao_v3$Previsto,col='blue',main='Real vs Previsto RF',pch=18, cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend='RF',pch=18,col='blue', bty='n', cex=.95)

plot(test_convert, previsoes_v4,col='green',main='Real vs Previsto NN',pch=18, cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend='NN',pch=18,col='green', bty='n', cex=.95)

plot(previsao_v2$Real, previsao_v2$Previsto,col='red',main='Real vs Previsto LM x RF x NN',pch=18,cex=0.7)
points(previsao_v3$Real, previsao_v3$Previsto,col='blue',pch=18,cex=0.7)
points(test_convert, previsoes_v4,col='green',pch=18,cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend=c('LM','RF','NN'),pch=18,col=c('red','blue','green'))

# Observe que conseguimos aumentar o nível de precisão do nosso modelo de 71% para 94%,
# e também, diminuir consideravelmente nosso erro médio. Sendo assim, podemos confirmar
# que neste caso, o modelo 3 (Redes Neurais) é melhor e mais estável que os modelos 1 e 2.



