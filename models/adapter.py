# import torch


# class EmotionAdapter(torch.nn.Module):
#     def __init__(self, num_emotions):
#         super().__init__()
#         self.emb = torch.nn.Embedding(num_emotions, 256)
#         self.conv = torch.nn.Sequential(
#             torch.nn.Conv1d(768 + 256, 512, kernel_size=3, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.Conv1d(512, 80, kernel_size=3, padding=1) # 80 — это количество Mel-bands
#         )

#     def forward(self, content_embeds, emotion_id):
#         # content_embeds: [B, T, 768]
#         # emotion_id: [B]
#         e = self.emb(emotion_id).unsqueeze(1).repeat(1, content_embeds.size(1), 1)
#         x = torch.cat([content_embeds, e], dim=-1)
#         x = x.transpose(1, 2) # Для Conv1d
#         return self.conv(x) # Выход: Мел-спектрограмма


# 1 Model. Linear
# import torch.nn as nn
# import torch.nn.functional as F

# class EmotionAdapter(nn.Module):
#     def __init__(self, input_dim=768, output_dim=100):
#         super().__init__()
#         self.map = nn.Sequential(
#             nn.Linear(input_dim, 512),
#             nn.LeakyReLU(0.2),
#             nn.Linear(512, output_dim)
#         )

#     def forward(self, x, target_len):
#         x = self.map(x)
#         x = x.transpose(1, 2) # [B, T, 100] -> [B, 100, T]
#         x = F.interpolate(x, size=target_len, mode='linear', align_corners=False)
#         return x


# 2 Model. Residual
# import torch.nn as nn
# import torch.nn.functional as F

# class ResidualConvBlock(nn.Module):
#     def __init__(self, channels, kernel_size=5, dilation=1):
#         super().__init__()
#         # Жесткий расчет паддинга, чтобы временная ось T не уменьшалась при свертке
#         padding = (kernel_size - 1) * dilation // 2
        
#         self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, 
#                                padding=padding, dilation=dilation)
#         self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, 
#                                padding=padding, dilation=dilation)
#         self.activation = nn.LeakyReLU(0.2)

#     def forward(self, x):
#         residual = x
#         x = self.conv1(x)
#         x = self.activation(x)
#         x = self.conv2(x)
#         # Прокидываем градиент напрямую — защита от затухания в глубоких слоях
#         return self.activation(x + residual)

# class EmotionAdapter(nn.Module):
#     def __init__(self, input_dim=768, hidden_dim=256, output_dim=100):
#         super().__init__()
        
#         # 1. Первичная поканальная проекция HuBERT признаков в скрытые каналы
#         self.input_proj = nn.Linear(input_dim, hidden_dim)
        
#         # 2. Сверточный стек с нарастающим Dilation (увеличиваем поле зрения модели)
#         # Блок 1 (dilation=1) видит локальные фонемы
#         # Блок 2 (dilation=2) и Блок 3 (dilation=4) связывают целые слоги и интонацию
#         self.conv_blocks = nn.Sequential(
#             ResidualConvBlock(hidden_dim, kernel_size=5, dilation=1),
#             ResidualConvBlock(hidden_dim, kernel_size=5, dilation=2),
#             ResidualConvBlock(hidden_dim, kernel_size=5, dilation=4)
#         )
        
#         # 3. Выходной слой: сжимаем скрытые каналы до 100 частотных бинов вокодера
#         self.output_proj = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)

#     def forward(self, x, target_len):
#         """
#         x: [B, T_hub, 768] — взвешенные эмбеддинги после пулинга
#         target_len: int — длина оригинальной мел-граммы для апсемплинга
#         """
#         # Сжимаем признаки по каналам: [B, T_hub, 768] -> [B, T_hub, 256]
#         x = self.input_proj(x)
        
#         # Переставляем оси для Conv1d: [B, T_hub, 256] -> [B, 256, T_hub]
#         x = x.transpose(1, 2)
        
#         # Напитываем кадры контекстом соседей
#         x = self.conv_blocks(x)
        
#         # Переводим в 100 каналов Vocos Mel: [B, 100, T_hub]
#         x = self.output_proj(x)
        
#         # Только теперь растягиваем время линейно до таргета
#         x = F.interpolate(x, size=target_len, mode='linear', align_corners=False)
        
#         return x # [B, 100, T_mel]


# 3.PostNet
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class ResidualConvBlock(nn.Module):
#     def __init__(self, channels, kernel_size=5, dilation=1):
#         super().__init__()
#         padding = (kernel_size - 1) * dilation // 2
        
#         self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, 
#                                padding=padding, dilation=dilation)
#         # Наш щит против взрыва градиентов — нормализация по оси времени
#         self.norm1 = nn.InstanceNorm1d(channels)
        
#         self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, 
#                                padding=padding, dilation=dilation)
#         self.norm2 = nn.InstanceNorm1d(channels)
        
#         self.activation = nn.LeakyReLU(0.2)

#     def forward(self, x):
#         residual = x
#         x = self.conv1(x)
#         x = self.norm1(x)
#         x = self.activation(x)
        
#         x = self.conv2(x)
#         x = self.norm2(x)
        
#         return self.activation(x + residual)

# class EmotionAdapter(nn.Module):
#     def __init__(self, input_dim=768, hidden_dim=256, output_dim=100):
#         super().__init__()
        
#         # ==========================================
#         # 1. PRE-NET (Извлечение контекста)
#         # ==========================================
#         self.input_proj = nn.Linear(input_dim, hidden_dim)
        
#         self.pre_blocks = nn.Sequential(
#             ResidualConvBlock(hidden_dim, kernel_size=5, dilation=1),
#             ResidualConvBlock(hidden_dim, kernel_size=5, dilation=2),
#             ResidualConvBlock(hidden_dim, kernel_size=5, dilation=4)
#         )
#         self.pre_to_mel = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)
        
#         # ==========================================
#         # 2. POST-NET (Наведение резкости гармоник)
#         # ==========================================
#         # Классический стек Tacotron2: 5 слоев сверток с большими ядрами
#         post_layers = []
#         post_channels = 512
        
#         # Первый слой Post-Net принимает 100 каналов мела
#         post_layers.append(nn.Sequential(
#             nn.Conv1d(output_dim, post_channels, kernel_size=5, padding=2),
#             nn.InstanceNorm1d(post_channels),
#             nn.Tanh()
#         ))
        
#         # 3 промежуточных слоя
#         for _ in range(3):
#             post_layers.append(nn.Sequential(
#                 nn.Conv1d(post_channels, post_channels, kernel_size=5, padding=2),
#                 nn.InstanceNorm1d(post_channels),
#                 nn.Tanh()
#             ))
            
#         # Финальный слой возвращает обратно в 100 каналов БЕЗ активации
#         post_layers.append(
#             nn.Conv1d(post_channels, output_dim, kernel_size=5, padding=2)
#         )
        
#         self.post_net = nn.Sequential(*post_layers)

#     def forward(self, x, target_len):
#         # 1. Прогон через Pre-Net в низком разрешении HuBERT
#         x = self.input_proj(x)
#         x = x.transpose(1, 2) # [B, hidden_dim, T_hub]
#         x = self.pre_blocks(x)
        
#         # Получаем грубый набросок мелграммы [B, 100, T_hub]
#         mel_coarse = self.pre_to_mel(x)
        
#         # 2. Апсемплинг времени до разрешения Vocos
#         mel_coarse = F.interpolate(mel_coarse, size=target_len, mode='linear', align_corners=False)
        
#         # 3. Наведение резкости через Post-Net
#         # Физика: Post-Net предсказывает остаточный шум/детали спектра
#         residual_error = self.post_net(mel_coarse)
        
#         # Итоговая мелграмма — сумма грубого скелета и резких деталей
#         mel_final = mel_coarse + residual_error
        
#         return mel_final # [B, 100, T_mel]


# # Model 4. 
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class ResidualConvBlock(nn.Module):
#     def __init__(self, channels, kernel_size=5, dilation=1):
#         super().__init__()
#         padding = (kernel_size - 1) * dilation // 2
        
#         self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, 
#                                padding=padding, dilation=dilation)
#         self.norm1 = nn.InstanceNorm1d(channels)
#         self.activation = nn.LeakyReLU(0.2)
        
#         self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, 
#                                padding=padding, dilation=dilation)
#         self.norm2 = nn.InstanceNorm1d(channels)

#     def forward(self, x):
#         residual = x
#         x = self.conv1(x)
#         x = self.norm1(x)
#         x = self.activation(x)
        
#         x = self.conv2(x)
#         x = self.norm2(x)
        
#         # Физика: Складываем ДО финальной активации, чтобы не копить сдвиг распределения
#         return self.activation(x + residual)

# class EmotionAdapter(nn.Module):
#     def __init__(self, input_dim=768, hidden_dim=256, output_dim=100):
#         super().__init__()
        
#         # 1. PRE-NET (Сжатые каналы для контроля емкости)
#         self.input_proj = nn.Linear(input_dim, hidden_dim)
        
#         self.pre_blocks = nn.Sequential(
#             ResidualConvBlock(hidden_dim, kernel_size=5, dilation=1),
#             ResidualConvBlock(hidden_dim, kernel_size=5, dilation=2),
#             ResidualConvBlock(hidden_dim, kernel_size=5, dilation=4)
#         )
#         self.pre_to_mel = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)
        
#         # 2. POST-NET (Сушка параметров: 512 -> 256, убрали InstanceNorm1d)
#         post_layers = []
#         post_channels = 256 
        
#         # Первый слой
#         post_layers.append(nn.Sequential(
#             nn.Conv1d(output_dim, post_channels, kernel_size=5, padding=2),
#             nn.Tanh(),
#             nn.Dropout(0.1) # Вместо InstanceNorm для борьбы с оверфитом
#         ))
        
#         # 3 промежуточных слоя
#         for _ in range(3):
#             post_layers.append(nn.Sequential(
#                 nn.Conv1d(post_channels, post_channels, kernel_size=5, padding=2),
#                 nn.Tanh(),
#                 nn.Dropout(0.1)
#             ))
            
#         # Финальный слой предсказания остатка
#         post_layers.append(
#             nn.Conv1d(post_channels, output_dim, kernel_size=5, padding=2)
#         )
        
#         self.post_net = nn.Sequential(*post_layers)

#     def forward(self, x, target_len):
#         x = self.input_proj(x)
#         x = x.transpose(1, 2) 
#         x = self.pre_blocks(x)
        
#         mel_coarse = self.pre_to_mel(x)
        
#         # Апсемплинг времени до разрешения Vocos
#         mel_coarse = F.interpolate(mel_coarse, size=target_len, mode='linear', align_corners=False)
        
#         # Post-Net теперь считает дельту без уничтожения масштаба на тесте
#         residual_error = self.post_net(mel_coarse)
        
#         mel_final = mel_coarse + residual_error
        
#         return mel_final


# 5 Model. 24.06.2026 prenet + postnet tipa
import torch
import torch.nn as nn
import torch.nn.functional as F

# Кастомный LayerNorm для сверток [B, C, T]. Нормализует каналы, сохраняя динамику времени!
class ChannelLayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(channels, eps=eps)
    def forward(self, x):
        # x: [B, C, T] -> перекидываем каналы в конец для LayerNorm
        x = x.transpose(1, 2)
        x = self.ln(x)
        return x.transpose(1, 2)

class ResidualConvBlock(nn.Module):
    def __init__(self, channels, kernel_size=5, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.norm1 = ChannelLayerNorm(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.norm2 = ChannelLayerNorm(channels)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, mask=None):
        residual = x
        x = self.conv1(x)
        if mask is not None:
            x = x * mask
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        if mask is not None:
            x = x * mask
        x = self.norm2(x)
        out = self.activation(x + residual)
        if mask is not None:
            out = out * mask
        return out


class EmotionAdapter(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=100):
        super().__init__()
        
        # 1. PRE-NET (Контекст времени на сетке HuBERT)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        # self.pre_blocks = nn.Sequential(
        #     ResidualConvBlock(hidden_dim, kernel_size=5, dilation=1),
        #     ResidualConvBlock(hidden_dim, kernel_size=5, dilation=2),
        #     ResidualConvBlock(hidden_dim, kernel_size=5, dilation=4)
        # )
        self.pre_blocks = nn.ModuleList([
            ResidualConvBlock(hidden_dim, kernel_size=5, dilation=1),
            ResidualConvBlock(hidden_dim, kernel_size=5, dilation=2),
            ResidualConvBlock(hidden_dim, kernel_size=5, dilation=4)
        ])
        self.pre_to_mel = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)
        
        # 2. POST-NET (Наведение резкости гармоник на сетке Вокодера)
        post_channels = 512
        post_layers = []
        
        # Входной слой Post-Net принимает 100 каналов грубого мела
        post_layers.append(nn.Sequential(
            nn.Conv1d(output_dim, post_channels, kernel_size=5, padding=2),
            ChannelLayerNorm(post_channels),
            nn.LeakyReLU(0.2)
        ))
        
        # 3 промежуточных слоя
        for _ in range(3):
            post_layers.append(nn.Sequential(
                nn.Conv1d(post_channels, post_channels, kernel_size=5, padding=2),
                ChannelLayerNorm(post_channels),
                nn.LeakyReLU(0.2)
                # Убрали жесткий Tanh, чтобы Post-Net мог дотягиваться до масштаба логарифма -16
            ))
            
        # Финал возвращает обратно в 100 каналов БЕЗ активации
        post_layers.append(
            nn.Conv1d(post_channels, output_dim, kernel_size=5, padding=2)
        )
        self.post_net = nn.Sequential(*post_layers)

    def forward(self, x, target_len, mask=None, mel_mask=None):
        '''
            x: [B, T_hub, 768]
            target_len: T_mel
            mask: [B, T_hub]
            mel_mask: [B, 100, T_mel]
        '''
        # 1. Pre-Net
        x = self.input_proj(x) # [B, T_hub, hidden_dim]
        x = x.transpose(1, 2) # [B, hidden_dim, T_hub] for Conv1d
        
        if mask is not None:
            mask = mask.unsqueeze(1) # [B, 1, T_hub]
            x = x * mask
            
            for block in self.pre_blocks:
                x = block(x, mask=mask)
            # x = self.pre_blocks(x) * mask 
            
            mel_coarse = self.pre_to_mel(x) * mask # [B, 100, T_hub]

            # Upsampling part
            B, C, _ = mel_coarse.shape
            # [B, 100, T_mel]
            mel_interpolated = torch.zeros((B, C, target_len), dtype=mel_coarse.dtype, device=mel_coarse.device)

            for i in range(B):
                real_hub_len = int(mask[i, 0].sum().item())
                real_mel_len = int(mel_mask[i].sum().item()) if mel_mask is not None else target_len
                pure_signal = mel_coarse[i:i+1, :, :real_hub_len] # [1, 100, real_hub_len]
                pure_interpolated = F.interpolate(pure_signal, size=real_mel_len, mode='nearest')
                mel_interpolated[i:i+1, :, :real_mel_len] = pure_interpolated

            mel_coarse = mel_interpolated

        else:
            # x = self.pre_blocks(x)
            for block in self.pre_blocks:
                x = block(x)
            mel_coarse = self.pre_to_mel(x) # [B, 100, T_hub]

            # Upsampling part
            mel_coarse = F.interpolate(mel_coarse, size=target_len, mode='nearest') # [B, 100, T_mel]
    
        # 2. Upsampling
        # mel_coarse = F.interpolate(mel_coarse, size=target_len, mode='nearest') # [B, 100, T_mel]
        
        # 3. Post-Net Residual
        if mel_mask is not None:
            mel_mask = mel_mask.unsqueeze(1)
            residual_error = self.post_net(mel_coarse) * mel_mask
            return (mel_coarse + residual_error) * mel_mask

        residual_error = self.post_net(mel_coarse)
        return mel_coarse + residual_error
