from AutoContext import AutoUnet

model = AutoUnet(4, 100)

model.load_stage1('')

model.train_stage2()

model.train_stage3()

model.train_end_to_end('', '')