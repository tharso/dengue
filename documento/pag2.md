Página 2: A Ciência dos Dados - Entendendo a Dinâmica do Clima e da Dengue
[Título Principal]
O Motor da Previsão: Cruzando Clima e Saúde Pública
Para construir um modelo capaz de prever surtos com precisão, precisamos de uma base sólida. Nosso projeto não trabalha com achismos; ele é fundamentado em uma década de dados históricos rigorosamente processados.
1. Nossas Fontes de Dados
A inteligência do nosso modelo nasce do cruzamento de duas realidades:
•	O Cenário Epidemiológico (DATASUS): Coletamos o histórico completo de casos de dengue no município de Presidente Prudente (SP) entre 2014 e 2024, analisados semana a semana (semanas epidemiológicas).
•	O Comportamento Climático (INMET): Integramos dados meteorológicos horários do Instituto Nacional de Meteorologia para o mesmo período de 10 anos. Isso inclui temperatura, umidade, precipitação e pressão atmosférica.
Os dados climáticos foram minuciosamente tratados para preencher lacunas e ajustados em médias semanais para sincronizar perfeitamente com os registros médicos da dengue.
2. Análise de Componentes Principais (PCA): Encontrando a Agulha no Palheiro
Com milhares de pontos de dados, como saber o que realmente importa para a proliferação do mosquito? Utilizamos a Análise de Componentes Principais (PCA), uma técnica estatística avançada que reduz a complexidade dos dados e revela as variáveis de maior impacto.
Nosso estudo isolou os 5 principais componentes (PCs) que ditam as regras do jogo:
•	O Peso do Longo Prazo (PC1): Descobrimos que a temperatura do ponto de orvalho e a chuva acumulada ao longo de várias semanas (especialmente entre 4 e 8 semanas anteriores) são os fatores mais fortes. Isso prova que o clima de semanas atrás dita o surto de hoje.
•	O Choque Térmico (PC2): Flutuações marcantes nas temperaturas (máxima, média e mínima) e na pressão atmosférica ditam o ritmo diário e sazonal do vetor.
•	Eventos Extremos (PC3): Variações abruptas, como secas intensas ou chuvas torrenciais repentinas (medidas pela precipitação total e umidade mínima), funcionam como gatilhos rápidos para a doença.
•	Vento e Umidade (PC4 e PC5): Nuances sutis no comportamento dos ventos e da umidade relativa completam o quebra-cabeça, mostrando como até mesmo a ventilação afeta a proliferação do Aedes aegypti.
3. Por Que Isso Importa?
Essas descobertas sublinham algo crucial: não basta olhar se choveu ontem. É a integração dessas análises climáticas detalhadas e multidimensionais com o machine learning que cria ferramentas preditivas verdadeiramente eficazes.

Tabela de Imagens e Prompts para a Página 2 (Metodologia e Dados)
Seção	Localização e Descrição	Prompt para a IA (Inglês)
1. Header (O Motor da Previsão)	Topo da página. Uma imagem conceitual que une o clima e a saúde, mostrando o cruzamento de informações globais.	A hyper-realistic, 3D conceptual illustration for a modern tech website header. A glowing, semi-transparent digital Earth globe focusing on Brazil. Surrounding the globe are elegant, glowing digital data streams. Some streams contain subtle climate icons (clouds, rain, thermometers), while others contain health data patterns. The streams merge together in a bright, energetic core. Dark background with cinematic neon blue and orange lighting. 8k resolution, highly detailed, professional.
2. Nossas Fontes de Dados	Ao lado do texto sobre DATASUS e INMET. Focada na magnitude de processar 10 anos de dados históricos.	A conceptual, futuristic server room or data center. Holographic displays float in the air, showing massive, intricate time-series graphs spanning a 10-year timeline. The graphs compare climate curves with data peaks. A professional data scientist, viewed from behind in silhouette, is observing and analyzing the complex data sets. High-tech environment, soft cyan and deep blue lighting, shallow depth of field, 8k, photorealistic.
3. Análise PCA	Ao lado da explicação sobre a redução de dimensionalidade e os componentes. Representação visual de extrair clareza do caos dos dados.	A highly detailed, macro-photography style conceptual image of data sorting. A glowing, intricate digital prism is taking a chaotic storm of scattered, multicolored glowing particles (representing raw complex data) and organizing them into five clean, distinct, brightly lit straight beams of light (representing the 5 Principal Components). Dark, moody background. Technological and mathematical aesthetic, sharp focus, 8k resolution, volumetric lighting.


import pandas as pd
import numpy as np

# Estruturando os dados da tabela extraída do artigo
dados_pca = {
    "VARIÁVEL": [
        "Precipitação Total (mm)", 
        "Pressão Atmosférica (mB)", 
        "Temperatura Ponto Orvalho (°C)",
        "Temperatura Máxima (°C)", 
        "Temperatura Média (°C)", 
        "Temperatura Mínima (°C)",
        "Umidade Relativa Média", 
        "Umidade Relativa Mínima", 
        "Vento Rajada Máxima (m/s)",
        "Vento Velocidade Média (m/s)", 
        "Acumulado 2 Semanas (mm)", 
        "Acumulado 3 Semanas (mm)",
        "Acumulado 4 Semanas (mm)", 
        "Acumulado 5 Semanas (mm)", 
        "Acumulado 6 Semanas (mm)",
        "Acumulado 7 Semanas (mm)", 
        "Acumulado 8 Semanas (mm)"
    ],
    "PC1": [0.537135, np.nan, 0.705313, np.nan, np.nan, np.nan, 0.628048, 0.572170, np.nan, np.nan, 0.686189, 0.781539, 0.844924, 0.889220, 0.917733, 0.936619, 0.947250],
    "PC2": [np.nan, 0.743673, np.nan, -0.926880, -0.925275, -0.826513, np.nan, np.nan, -0.530554, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    "PC3": [-0.609034, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, -0.508900, np.nan, np.nan, -0.520941, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    "PC4": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, -0.511273, -0.510786, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    "PC5": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, -0.617125, -0.823443, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
}

# Criando o DataFrame
df_pca = pd.DataFrame(dados_pca)

# Exibindo a tabela
print(df_pca.to_string(index=False))