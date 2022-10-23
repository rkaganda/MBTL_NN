# import model
# import torch
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# my_model, opti = model.setup_model(
#     frames_per_observation=1,
#     input_state_size=1,
#     state_state_size=1,
#     learning_rate=1
# )
#
# in_tensor = torch.ones(3).to(device)
# my_model.to(device)
# print(my_model)
#
# print(in_tensor)
#
# out_tensor = my_model(in_tensor)
#
# print(out_tensor)

delay = 20

window = 20

last = 41

a = [n for n in range(0, last)]

print(a[-1])

print(a[:-delay][-window:])

print(45-delay-1)
print(45-delay-window)