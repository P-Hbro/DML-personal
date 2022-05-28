import torch
import model
start_evt = torch.cuda.Event(enable_timing=True)
end_evt = torch.cuda.Event(enable_timing=True)
start_evt.record()

# start the event: training, communicating, etc.
model.train_parallel()

end_evt.record()
torch.cuda.synchronize()

whole_time = start_evt.elapsed_time(end_evt)
