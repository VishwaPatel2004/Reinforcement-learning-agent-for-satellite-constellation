from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether, Flowable, Image, NextPageTemplate,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "pdf" / "Satellite_Safe_PPO_Presentation_Report.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

NAVY = colors.HexColor("#07182E")
BLUE = colors.HexColor("#1261A0")
CYAN = colors.HexColor("#1DB4D7")
GREEN = colors.HexColor("#27AE60")
AMBER = colors.HexColor("#F4B942")
RED = colors.HexColor("#D94B4B")
INK = colors.HexColor("#182433")
MUTED = colors.HexColor("#58697A")
PALE = colors.HexColor("#EEF5FA")
WHITE = colors.white

font_regular = "Helvetica"
font_bold = "Helvetica-Bold"
for path in [Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/calibri.ttf")]:
    if path.exists():
        pdfmetrics.registerFont(TTFont("BodyFont", str(path)))
        font_regular = "BodyFont"
        break
for path in [Path("C:/Windows/Fonts/arialbd.ttf"), Path("C:/Windows/Fonts/calibrib.ttf")]:
    if path.exists():
        pdfmetrics.registerFont(TTFont("BodyBold", str(path)))
        font_bold = "BodyBold"
        break

styles = getSampleStyleSheet()
TITLE = ParagraphStyle("TitleX", fontName=font_bold, fontSize=29, leading=34, textColor=WHITE, alignment=TA_LEFT)
SUBTITLE = ParagraphStyle("SubTitleX", fontName=font_regular, fontSize=13, leading=19, textColor=colors.HexColor("#CFE7F5"))
H1 = ParagraphStyle("H1X", fontName=font_bold, fontSize=22, leading=27, textColor=NAVY, spaceAfter=10)
H2 = ParagraphStyle("H2X", fontName=font_bold, fontSize=14, leading=18, textColor=BLUE, spaceBefore=7, spaceAfter=5)
BODY = ParagraphStyle("BodyX", fontName=font_regular, fontSize=9.3, leading=13.2, textColor=INK, spaceAfter=6)
BULLET = ParagraphStyle("BulletX", parent=BODY, leftIndent=12, firstLineIndent=-7, bulletIndent=0, spaceAfter=4)
SMALL = ParagraphStyle("SmallX", fontName=font_regular, fontSize=7.8, leading=10.5, textColor=MUTED)
CALLOUT = ParagraphStyle("CalloutX", fontName=font_bold, fontSize=10.5, leading=15, textColor=NAVY, alignment=TA_CENTER)
CODE = ParagraphStyle("CodeX", fontName="Courier", fontSize=7.5, leading=10, textColor=INK, backColor=colors.HexColor("#F4F7F9"), borderPadding=7)


class WorkflowDiagram(Flowable):
    def __init__(self, width=175*mm, height=104*mm):
        super().__init__(); self.width = width; self.height = height

    def draw(self):
        c = self.canv
        steps = [
            ("1", "Orbital data", "OMM records\nStarlink + debris", BLUE),
            ("2", "Initial propagation", "Skyfield converts\nelements to r and v", CYAN),
            ("3", "RL observation", "427 values: satellite\nstates + risk + fuel", colors.HexColor("#6C63B5")),
            ("4", "PPO policy", "Actor proposes 210\ncontinuous controls", GREEN),
            ("5", "Environment step", "Update motion, fuel,\ncoverage and distance", AMBER),
            ("6", "Reward + safety", "Task reward minus\nadaptive risk penalty", RED),
            ("7", "PPO update", "Clipped policy loss +\nvalue-function learning", BLUE),
            ("8", "Evaluation", "Fixed seeds, collisions,\ncoverage, fuel, return", GREEN),
        ]
        cols, bw, bh, gx, gy = 2, 73*mm, 18*mm, 24*mm, 7*mm
        x0, y0 = 4*mm, self.height - bh - 4*mm
        coords = []
        for i, (num, title, desc, col) in enumerate(steps):
            row, col_idx = divmod(i, cols)
            if row % 2 == 1: col_idx = 1 - col_idx
            x = x0 + col_idx * (bw + gx)
            y = y0 - row * (bh + gy)
            coords.append((x, y))
            c.setFillColor(colors.white); c.setStrokeColor(col); c.setLineWidth(1.5)
            c.roundRect(x, y, bw, bh, 4*mm, fill=1, stroke=1)
            c.setFillColor(col); c.circle(x + 8*mm, y + bh/2, 5.2*mm, fill=1, stroke=0)
            c.setFillColor(WHITE); c.setFont(font_bold, 9); c.drawCentredString(x+8*mm, y+bh/2-3, num)
            c.setFillColor(NAVY); c.setFont(font_bold, 9); c.drawString(x+16*mm, y+bh-6*mm, title)
            c.setFillColor(MUTED); c.setFont(font_regular, 6.8)
            for j, line in enumerate(desc.split("\n")):
                c.drawString(x+16*mm, y+bh-10.3*mm-j*3.3*mm, line)
        c.setStrokeColor(MUTED); c.setFillColor(MUTED); c.setLineWidth(1.2)
        for i in range(len(coords)-1):
            x1, y1 = coords[i]; x2, y2 = coords[i+1]
            if abs(y1-y2) < 1:
                if x2 > x1: a=(x1+bw, y1+bh/2); b=(x2, y2+bh/2)
                else: a=(x1, y1+bh/2); b=(x2+bw, y2+bh/2)
            else:
                a=(x1+bw/2, y1); b=(x2+bw/2, y2+bh)
            c.line(a[0], a[1], b[0], b[1])
            c.line(b[0], b[1], b[0]-2.2*mm, b[1]+2.2*mm)
            c.line(b[0], b[1], b[0]+2.2*mm, b[1]+2.2*mm)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(NAVY); canvas.rect(0, 0, A4[0], 12*mm, fill=1, stroke=0)
    canvas.setFillColor(colors.HexColor("#CFE7F5")); canvas.setFont(font_regular, 7.5)
    canvas.drawString(18*mm, 5*mm, "Satellite Collision Avoidance | Safe PPO Presentation Guide")
    canvas.drawRightString(A4[0]-18*mm, 5*mm, f"{doc.page}")
    canvas.restoreState()


def cover(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(NAVY); canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
    canvas.setFillColor(BLUE); canvas.circle(A4[0]-18*mm, A4[1]-28*mm, 48*mm, fill=1, stroke=0)
    canvas.setFillColor(CYAN); canvas.circle(A4[0]-18*mm, A4[1]-28*mm, 34*mm, fill=1, stroke=0)
    canvas.setStrokeColor(colors.HexColor("#8EDAF0")); canvas.setLineWidth(0.6)
    for r in [58, 72, 88]: canvas.circle(A4[0]-18*mm, A4[1]-28*mm, r*mm, fill=0, stroke=1)
    canvas.setFillColor(WHITE); canvas.circle(A4[0]-66*mm, A4[1]-57*mm, 2.2*mm, fill=1, stroke=0)
    canvas.restoreState()


doc = BaseDocTemplate(str(OUT), pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=17*mm, bottomMargin=18*mm,
                      title="Satellite Collision Avoidance - Safe PPO Presentation Report", author="Keyur Pandya")
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
doc.addPageTemplates([PageTemplate(id="cover", frames=frame, onPage=cover), PageTemplate(id="body", frames=frame, onPage=footer)])

story = []
story += [Spacer(1, 74*mm), Paragraph("Satellite Collision Avoidance<br/>Using Safe PPO", TITLE), Spacer(1, 6*mm),
          Paragraph("A presentation-ready guide to the complete project, data flow, reinforcement-learning environment, adaptive safety penalty, training loop, evaluation, dashboard, and limitations.", SUBTITLE),
          Spacer(1, 15*mm), Paragraph("70 controlled satellites  |  up to 500 debris objects  |  427-D state  |  210-D continuous action", ParagraphStyle("coverstat", parent=SMALL, textColor=WHITE, fontSize=9)),
          Spacer(1, 29*mm), Paragraph("Prepared for project explanation and viva", ParagraphStyle("coverfoot", parent=BODY, textColor=colors.HexColor("#CFE7F5"))), NextPageTemplate("body"), PageBreak()]

def heading(title, kicker=None):
    if kicker: story.append(Paragraph(kicker.upper(), ParagraphStyle("kick"+str(len(story)), parent=SMALL, textColor=CYAN, fontName=font_bold)))
    story.append(Paragraph(title, H1))

def bullets(items):
    for x in items: story.append(Paragraph("• " + x, BULLET))

def callout(text, color=PALE):
    t=Table([[Paragraph(text, CALLOUT)]], colWidths=[doc.width], rowHeights=None)
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),color),("BOX",(0,0),(-1,-1),1,CYAN),("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10),("TOPPADDING",(0,0),(-1,-1),9),("BOTTOMPADDING",(0,0),(-1,-1),9)])); story.append(t); story.append(Spacer(1,6))

heading("1. The project in one minute", "Opening explanation")
story.append(Paragraph("The project studies how a reinforcement-learning controller can coordinate a satellite constellation while reducing proximity risk from tracked debris. It combines orbital-element data, Skyfield initialization, a custom Gymnasium environment, PyTorch vectorized calculations, Stable-Baselines3 training, quantitative evaluation, and a Plotly dashboard.", BODY))
callout("One-sentence pitch: An RL agent observes the constellation and nearby debris, proposes small control actions, and learns a balance between operational reward, fuel use, and a dynamically weighted collision-risk penalty.")
story.append(Paragraph("Problem formulation", H2))
data=[["Element","Project choice","Why it matters"],["Controlled objects","70 Starlink records","High-dimensional coordinated control"],["Debris set","Up to 500 records","35,000 pairwise distances each step"],["Decision interval","10 seconds","Repeated closed-loop decisions"],["Episode","540 steps / 90 minutes","Approximately one LEO-orbit timescale"],["Primary method","PPO + adaptive risk penalty","Stable continuous-control learning"]]
t=Table([[Paragraph(str(c), BODY if r else ParagraphStyle("th",parent=BODY,textColor=WHITE,fontName=font_bold)) for c in row] for r,row in enumerate(data)], colWidths=[34*mm,48*mm,88*mm], repeatRows=1)
t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),BLUE),("GRID",(0,0),(-1,-1),.35,colors.HexColor("#B8C8D5")),("VALIGN",(0,0),(-1,-1),"TOP"),("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE,PALE]),("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)])); story.append(t)
story.append(Paragraph("Presentation tip", H2)); bullets(["Begin with the operational conflict: maximize service while avoiding debris and conserving fuel.","Say that the current system is a research prototype, not a flight-certified collision-avoidance system.","Use 'adaptive Lagrangian-style penalty' rather than claiming a formal safety guarantee."])
story.append(PageBreak())

heading("2. Complete system workflow", "Main diagram")
story.append(WorkflowDiagram())
callout("Training repeats steps 3-7. Deployment or demonstration repeats steps 2-6 using a frozen trained policy; no gradient update occurs during inference.")
story.append(Paragraph("Three loops to explain", H2))
bullets(["Data loop: download orbital records, validate them, then generate initial positions and velocities.","Learning loop: observe, act, simulate, score, collect rollout, and update actor/critic parameters.","Evaluation loop: freeze the policy and compare methods under identical initial seeds and metrics."])
story.append(PageBreak())

heading("3. Data and orbital-state pipeline", "Input layer")
story.append(Paragraph("The downloaded JSON files contain OMM orbital-element records. At environment reset, Skyfield constructs an EarthSatellite object for every usable record and propagates it to a selected epoch. The resulting Cartesian position and velocity vectors initialize tensors used by the simulator.", BODY))
data=[["Stage","Input","Output"],["Download","CelesTrak catalog query","OMM JSON files"],["Selection","Object type and configured limits","70 satellites + up to 500 debris"],["Propagation","OMM record + epoch","r = [x,y,z] km; v = [vx,vy,vz] km/s"],["Tensor conversion","NumPy arrays","GPU/CPU float32 tensors"],["Distance engine","Satellite and debris positions","70 x 500 pairwise-distance matrix"]]
t=Table([[Paragraph(str(c), BODY if r else ParagraphStyle("th2",parent=BODY,textColor=WHITE,fontName=font_bold)) for c in row] for r,row in enumerate(data)],colWidths=[34*mm,65*mm,71*mm],repeatRows=1)
t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),BLUE),("GRID",(0,0),(-1,-1),.4,colors.HexColor("#BBCAD5")),("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE,PALE]),("VALIGN",(0,0),(-1,-1),"TOP"),("PADDING",(0,0),(-1,-1),6)]));story.append(t)
story.append(Paragraph("Why PyTorch is used", H2)); bullets(["torch.cdist computes all satellite-debris distances in one vectorized operation.","The same implementation can use CUDA when available and CPU otherwise.","Vectorization removes slow nested Python loops; actual speed must be measured on the presentation machine."])
story.append(Paragraph("Important distinction", H2)); callout("Skyfield supplies the initial physically meaningful state. The current training environment then uses a simplified linear motion model, so it should be described as an accelerated prototype simulator, not full SGP4 propagation after every maneuver.", colors.HexColor("#FFF4DB"))
story.append(PageBreak())

heading("4. MDP: what the agent sees and does", "Environment design")
story.append(Paragraph("A reinforcement-learning problem is defined by state, action, transition, reward, and termination. The custom Gymnasium environment exposes these through reset() and step().", BODY))
data=[["Observation component","Size","Meaning"],["70 positions and velocities","420","Six values per controlled satellite"],["Closest debris position","3","Coordinates of the globally closest debris object"],["Minimum distance","1","Normalized proximity indicator"],["Collision-risk score","1","Gaussian distance-based risk proxy"],["Coverage","1","Fraction of sampled ground grid marked covered"],["Fuel","1","Remaining normalized fuel budget"],["Total","427","Input to the policy and value networks"]]
t=Table([[Paragraph(str(c), BODY if r else ParagraphStyle("th3",parent=BODY,textColor=WHITE,fontName=font_bold)) for c in row] for r,row in enumerate(data)],colWidths=[62*mm,20*mm,88*mm],repeatRows=1)
t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),BLUE),("GRID",(0,0),(-1,-1),.35,colors.HexColor("#BBCAD5")),("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE,PALE]),("PADDING",(0,0),(-1,-1),5)]));story.append(t)
story.append(Paragraph("Action space", H2)); story.append(Paragraph("The action has 210 continuous values: three components for each of 70 satellites. Every value is clipped to the configured maximum magnitude. The environment accumulates their effect and charges a small fuel cost.", BODY))
callout("State: 427 values -> actor network -> action: 210 continuous values -> environment transition")
story.append(Paragraph("Episode ending", H2)); bullets(["Termination: modeled separation falls below 1 km.","Truncation: 540 steps are reached or fuel becomes zero.","The environment returns the next observation, reward, end flags, and diagnostic information."])
story.append(PageBreak())

heading("5. PPO: the learning engine", "Core algorithm")
story.append(Paragraph("PPO is an on-policy actor-critic method. The actor represents the thrust policy and the critic estimates the long-term value of a state. A rollout of 2,048 transitions is collected, advantages are calculated, and mini-batch optimization is repeated for 10 epochs.", BODY))
story.append(Paragraph("Clipped policy objective", H2))
story.append(Paragraph("L_clip = E[min(r_t A_t, clip(r_t, 1-epsilon, 1+epsilon) A_t)]", CODE))
bullets(["r_t compares the probability of the action under the new and old policies.","A_t is the estimated advantage: whether the action performed better than expected.","Clipping limits overly large policy changes and improves training stability.","The critic minimizes value-prediction error; entropy encourages exploration."])
data=[["Hyperparameter","Configured value","Meaning"],["Learning rate","3e-4","Optimizer step size"],["Gamma","0.99","Weight given to future reward"],["Rollout length","2,048","Transitions before an update"],["Batch size","64","Samples per gradient mini-batch"],["Epochs","10","Passes over each rollout"],["Training steps","50,000","Current experiment budget"]]
t=Table([[Paragraph(str(c),BODY if r else ParagraphStyle("th4",parent=BODY,textColor=WHITE,fontName=font_bold)) for c in row] for r,row in enumerate(data)],colWidths=[48*mm,35*mm,87*mm],repeatRows=1)
t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),BLUE),("GRID",(0,0),(-1,-1),.35,colors.HexColor("#BBCAD5")),("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE,PALE]),("PADDING",(0,0),(-1,-1),5)])); story.append(t)
story.append(Paragraph("What PPO learns", H2)); callout("It does not receive a hand-written avoidance direction. It adjusts policy parameters so that actions associated with higher long-term penalized return become more likely.")
story.append(PageBreak())

heading("6. What makes this implementation 'Safe PPO'", "Adaptive risk penalty")
story.append(Paragraph("The project augments the task reward with a collision-risk cost. A non-negative multiplier increases when risk exceeds the configured limit and decreases when risk stays below it. The current code performs this update at each environment step.", BODY))
story.append(Paragraph("Risk proxy", H2)); story.append(Paragraph("c_t = exp(-d_min^2 / (2 sigma^2)), with sigma = 5 km", CODE))
story.append(Paragraph("Multiplier update", H2)); story.append(Paragraph("lambda <- clip(lambda + alpha_lambda (c_t - cost_limit), 0, lambda_max)", CODE))
story.append(Paragraph("Effective learning reward", H2)); story.append(Paragraph("r_safe = r_coverage + r_distance + r_fuel - lambda c_t", CODE))
data=[["Condition","Multiplier response","Learning effect"],["Risk above limit","lambda increases","Future risky transitions receive a larger penalty"],["Risk below limit","lambda may decrease","The policy can focus more on coverage and fuel"],["Very close approach","Large distance/collision penalty","Strong negative training signal"],["Safe and efficient motion","Good task reward, low cost","Desired behavior is reinforced"]]
t=Table([[Paragraph(str(c),BODY if r else ParagraphStyle("th5",parent=BODY,textColor=WHITE,fontName=font_bold)) for c in row] for r,row in enumerate(data)],colWidths=[43*mm,51*mm,76*mm],repeatRows=1)
t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),GREEN),("GRID",(0,0),(-1,-1),.35,colors.HexColor("#BBCAD5")),("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE,colors.HexColor("#EFF9F3")]),("PADDING",(0,0),(-1,-1),6)])); story.append(t)
story.append(Spacer(1,7)); callout("Defensible description: PPO trained with an adaptive Lagrangian-style collision-risk penalty. Do not claim guaranteed safety unless a true constrained optimizer and formal assumptions are implemented and verified.", colors.HexColor("#FFF4DB"))
story.append(PageBreak())

heading("7. One complete training cycle", "Step-by-step narration")
steps=[("Reset","Choose a seeded time offset, propagate records, initialize tensors, fuel, and metrics."),("Observe","Create the 427-dimensional state vector."),("Act","The stochastic PPO policy samples 210 clipped continuous controls."),("Transition","Update simplified motion, distance matrix, coverage proxy, fuel, and risk."),("Score","Calculate task reward and subtract the adaptive safety penalty."),("Store","Save observation, action, reward, value estimate, log probability, and end flags."),("Estimate","Compute discounted returns and generalized advantages after the rollout."),("Optimize","Update actor and critic in shuffled mini-batches using the clipped PPO objective."),("Evaluate","Periodically run deterministic episodes and save the best model by evaluation reward."),("Export","Save final/checkpoint models and TensorBoard logs for later comparison.")]
rows=[]
for i,(a,b) in enumerate(steps,1): rows.append([Paragraph(str(i),ParagraphStyle("num"+str(i),parent=CALLOUT,textColor=WHITE)),Paragraph(a,ParagraphStyle("st"+str(i),parent=BODY,fontName=font_bold,textColor=NAVY)),Paragraph(b,BODY)])
t=Table(rows,colWidths=[10*mm,29*mm,131*mm])
t.setStyle(TableStyle([("BACKGROUND",(0,0),(0,-1),BLUE),("ROWBACKGROUNDS",(1,0),(-1,-1),[WHITE,PALE]),("GRID",(0,0),(-1,-1),.3,colors.HexColor("#BBCAD5")),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("PADDING",(0,0),(-1,-1),5)]));story.append(t)
story.append(Spacer(1,8)); story.append(Paragraph("Training vs inference", H2)); bullets(["Training includes exploration and gradient updates.","Inference normally uses deterministic model.predict() and does not modify weights.","Safety must be evaluated on unseen, controlled scenarios rather than inferred from training reward alone."])
story.append(PageBreak())

heading("8. Evaluation and fair comparison", "Evidence layer")
story.append(Paragraph("A convincing evaluation freezes every trained model and runs it on the same episode seeds. Report distributions, not only a single attractive run.", BODY))
data=[["Metric","What it answers","Recommended report"],["Collision rate","How often did termination occur?","count / episodes + confidence interval"],["Minimum separation","How close was the worst encounter?","median, minimum, percentile distribution"],["Mean risk cost","Was the risk budget respected?","mean and maximum per episode"],["Coverage","Was the service objective maintained?","mean and worst-case percentage"],["Fuel consumed","Was avoidance economical?","mean and distribution"],["Episode return","What did the coded objective optimize?","mean +/- standard deviation"]]
t=Table([[Paragraph(str(c),BODY if r else ParagraphStyle("th6",parent=BODY,textColor=WHITE,fontName=font_bold)) for c in row] for r,row in enumerate(data)],colWidths=[37*mm,65*mm,68*mm],repeatRows=1)
t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),BLUE),("GRID",(0,0),(-1,-1),.35,colors.HexColor("#BBCAD5")),("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE,PALE]),("PADDING",(0,0),(-1,-1),5)])); story.append(t)
story.append(Paragraph("Baselines", H2)); bullets(["No-thrust controller: establishes natural encounter behavior in the simplified simulator.","Random controller: establishes action-space difficulty.","Basic PPO: same environment and reward, with the adaptive safety term disabled.","DDPG, SAC, A2C, TD3, and TRPO: use identical data splits, seeds, steps, and metrics."])
callout("Current caution: step06_compare_algorithms.py contains hard-coded illustrative metrics. Replace them with actual model evaluations before calling the chart an experimental result.", colors.HexColor("#FDECEC"))
story.append(PageBreak())

heading("9. Dashboard and demonstration workflow", "Presentation layer")
story.append(Paragraph("The dashboard turns numerical state into an understandable story: Earth and orbital objects, nearest threats, risk status, coverage indicators, fuel, and action vectors. It should be presented as a visualization of the prototype controller, not as a live spacecraft operations system.", BODY))
data=[["Dashboard stage","What happens"],["Select mode","Choose no controller or a trained algorithm"],["Refresh state","Use stored orbital records and the current demonstration time"],["Compute proximity","Build the satellite-debris distance matrix"],["Predict action","Pass the observation to the selected frozen model"],["Update display","Show positions, risk table, metrics, and maneuver vectors"],["Explain outcome","Connect the action to separation, fuel, and coverage trade-offs"]]
t=Table([[Paragraph(str(c),BODY if r else ParagraphStyle("th7",parent=BODY,textColor=WHITE,fontName=font_bold)) for c in row] for r,row in enumerate(data)],colWidths=[49*mm,121*mm],repeatRows=1)
t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),BLUE),("GRID",(0,0),(-1,-1),.35,colors.HexColor("#BBCAD5")),("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE,PALE]),("PADDING",(0,0),(-1,-1),7)])); story.append(t)
story.append(Paragraph("Suggested live-demo narration", H2));
story.append(Paragraph("'Here the environment has converted the constellation into the same 427-value observation used during training. The selected policy produces three control components for every satellite. We then recompute proximity, the risk proxy, fuel, and coverage. The important behavior is not merely movement; it is whether risk decreases without excessive fuel use or service loss.'", ParagraphStyle("quote",parent=BODY,leftIndent=10,rightIndent=10,borderColor=CYAN,borderWidth=1,borderPadding=8,backColor=PALE)))
story.append(Paragraph("Before presenting", H2)); bullets(["Refresh the orbital data if you want to call it current; the bundled snapshots are not continuously live.","Pre-run the model load and dashboard once on the presentation computer.","Keep screenshots or a short fallback recording in case the live server fails."])
story.append(PageBreak())

heading("10. Limitations and improvement roadmap", "Questions the panel may ask")
data=[["Current prototype limitation","Why it matters","Next improvement"],["Linear post-reset motion","Does not capture full perturbed orbit dynamics","Propagate maneuvers with a validated orbital dynamics model"],["Action effect is simplified","Current accumulation is not a rigorous delta-v integration","Update velocity, then integrate position consistently"],["Coverage proxy saturates","May give 100% before learning","Use line-of-sight, elevation angle, and Earth occlusion"],["Risk is distance-derived","Not a calibrated probability of collision","Use conjunction geometry and covariance-based Pc"],["Global closest debris only","State omits threat assignment per satellite","Encode nearest threats per controlled object"],["Observation bounds violated","Can destabilize policy learning","Normalize/clip values and run check_env"],["Adaptive penalty in base env","Weakens safe-vs-basic comparison","Separate task reward and constraint wrappers"],["Limited evaluation","Can exaggerate lucky outcomes","Use held-out scenarios, confidence intervals, ablations"]]
t=Table([[Paragraph(str(c),SMALL if r else ParagraphStyle("th8",parent=SMALL,textColor=WHITE,fontName=font_bold)) for c in row] for r,row in enumerate(data)],colWidths=[48*mm,57*mm,65*mm],repeatRows=1)
t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),RED),("GRID",(0,0),(-1,-1),.35,colors.HexColor("#BBCAD5")),("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE,colors.HexColor("#FFF5F5")]),("VALIGN",(0,0),(-1,-1),"TOP"),("PADDING",(0,0),(-1,-1),5)]));story.append(t)
story.append(Spacer(1,7)); callout("A strong viva answer acknowledges limitations, explains why the prototype is still useful, and gives a concrete validation roadmap.")
story.append(PageBreak())

heading("11. Seven-minute presentation script", "Ready-to-speak structure")
script=[("0:00-0:40 - Problem","Space debris creates a safety-versus-service control problem for large constellations. My project explores whether RL can coordinate avoidance while considering coverage and fuel."),("0:40-1:20 - Data","I use orbital-element records for Starlink and major debris families. Skyfield converts them into Cartesian position and velocity states at reset."),("1:20-2:10 - Environment","The Gymnasium environment gives the agent 427 inputs and accepts 210 continuous controls. PyTorch calculates the full distance matrix efficiently."),("2:10-3:20 - PPO","PPO uses actor and critic networks. It gathers rollouts and applies a clipped objective so updates do not change the policy too aggressively."),("3:20-4:30 - Safety mechanism","I add a distance-based collision-risk cost and a non-negative adaptive multiplier. When risk is above the limit, the penalty becomes stronger; when risk is low, task performance receives more emphasis."),("4:30-5:20 - Training","Each cycle is observe, act, simulate, reward and cost, store rollout, estimate advantages, then optimize actor and critic."),("5:20-6:10 - Evaluation","Models must be frozen and tested on identical seeds. I report collision rate, minimum distance, coverage, fuel, cost, and return."),("6:10-7:00 - Demo and conclusion","The dashboard visualizes state, threats, actions, and metrics. This is a research prototype demonstrating the complete safe-RL workflow; future work is higher-fidelity dynamics and calibrated risk.")]
for a,b in script: story.append(KeepTogether([Paragraph(a,H2),Paragraph(b,BODY)]))
story.append(PageBreak())

heading("12. Viva questions and concise answers", "Final preparation")
qa=[("Why PPO?","It is stable, supports continuous actions, and its clipped update is simpler and less expensive than a hard trust-region method."),("Why is it called safe?","The training objective includes an explicit risk cost whose penalty weight adapts to constraint violation. In this implementation it is a safety-oriented penalty method, not a formal deployment guarantee."),("Why use a Lagrangian multiplier?","A fixed penalty can be too weak or unnecessarily strong. The multiplier responds to measured risk relative to the configured limit."),("What is the state dimension?","427: 420 satellite position/velocity values, 3 closest-debris coordinates, and 4 scalar metrics."),("What is the action dimension?","210: three continuous control components for each of 70 satellites."),("How is collision probability computed?","The current environment uses a Gaussian function of minimum separation as a risk proxy. A flight-grade system would incorporate covariance and conjunction geometry."),("How do you prove improvement?","Evaluate frozen models on the same held-out scenarios, report uncertainty, and perform an ablation with the adaptive safety term disabled."),("What is your main contribution?","An integrated prototype pipeline connecting real orbital-element initialization, vectorized constellation-scale proximity calculations, PPO training with adaptive risk penalization, comparative evaluation, and interactive visualization.")]
for q,a in qa:
    box=Table([[Paragraph("Q: "+q,ParagraphStyle("q"+str(len(story)),parent=BODY,fontName=font_bold,textColor=BLUE)),Paragraph("A: "+a,BODY)]],colWidths=[47*mm,123*mm])
    box.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),PALE),("BOX",(0,0),(-1,-1),.5,colors.HexColor("#B9CDD9")),("VALIGN",(0,0),(-1,-1),"TOP"),("PADDING",(0,0),(-1,-1),6)]));story.append(box);story.append(Spacer(1,4))
story.append(Spacer(1,5)); callout("Closing line: The value of the project is the complete decision loop - from orbital data to learned action to measurable safety and operational outcomes - plus a clear path toward higher-fidelity validation.")

doc.build(story)
print(OUT)
