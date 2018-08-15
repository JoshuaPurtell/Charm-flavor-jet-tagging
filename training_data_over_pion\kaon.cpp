//include C++
#include <chrono>
#include <random>
#include <iostream>
#include <vector>
#include "TRandom3.h"

//include ROOT
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TMath.h"
#include "TRandomGen.h"
#include "TTree.h"
#include "TLorentzVector.h"
#include "TClonesArray.h"
#include "fastjet/config.h"
#include "SystemOfUnits.h"

//include pythia
#include "Pythia8/Pythia.h"

//include fastjet
#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/ClusterSequenceArea.hh"
#include "fastjet/Selector.hh"
#include "fastjet/tools/Subtractor.hh"
#include "fastjet/tools/JetMedianBackgroundEstimator.hh"

//include STAR
#include "StPhysicalHelix.hh"

//set namespace
using namespace fastjet;
using namespace Pythia8;
using namespace std;

//initialize command-line editable variables
long long int nEvents;
TString fout_name;
int input_seed_multiple;

class RandNumGen {
    private:
    TRandomMixMax PRNG;
    std::hash<uint64_t> t_hash;
    float tof_eff;
    public:
    RandNumGen (int seed_multiplier) :
    PRNG{t_hash(static_cast<uint64_t>(time(nullptr)*seed_multiplier))}{};
    long long int rand_seed(){
    return PRNG.Integer(900000001)-1;
    };
    void say_it() { cout << PRNG.Uniform() << endl; };
    float rand() { return PRNG.Rndm(); }
};

//defining a class called MyInfo that inherits UserInfoBase
class MyInfo: public PseudoJet::UserInfoBase {
    public: MyInfo(int id, double xDec, double yDec, double zDec, double xp, double yp, double zp, double chrg, int moth1, int motherid) : _pdg_id(id), _x_Dec(xDec), _y_Dec(yDec), _z_Dec(zDec), _x_p(xp), _y_p(yp), _z_p(zp), _chrg_(chrg), _moth_1(moth1), _mother_id(motherid){};
  
    int pdg_id() const {return _pdg_id;}
    double x_Dec() const {return _x_Dec;}
    double y_Dec() const {return _y_Dec;}
    double z_Dec() const {return _z_Dec;}
    double x_p() const {return _x_p;}
    double y_p() const {return _y_p;}
    double z_p() const {return _z_p;}
    double chrg_() const {return _chrg_;}
    int moth_1() const {return _moth_1;}
    int mother_id() const {return _mother_id;}
  
    private:
    int _pdg_id;
    double _x_Dec;
    double _y_Dec;
    double _z_Dec;
    double _x_p;
    double _y_p;
    double _z_p;
    double _chrg_;
    int _moth_1;
    int _mother_id;
      
};

//ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//     M A I N
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
int main( int argc, const char** argv ){
    
    //creates command-line options
    if (argc == 2 && !strcmp(argv[1],"-h")){
    cout << "  * args are:   "                           << endl
         << "  * 1: nEvents              |  30     "     << endl
         << "  * 2: fout_name            |  out_train_PM.root   " << endl
         << "  * 3: input seed multiple  |  1   "        << endl;
    exit(0);
    }

    // default arguments (to be replaced by command line arguments)
    nEvents             = (argc > 1)  ? atoi(argv[1]) : 30;
    fout_name           = (argc > 2)  ? argv[2]        : "out_train_PM.root";
    input_seed_multiple = (argc > 3)  ? atoi(argv[3])  : 1;
    
    //Constructs pythia 
    RandNumGen prng(input_seed_multiple);
    if (!fout_name.EndsWith(".root")) fout_name.Append(".root");
    long long int times_failed = 0;
    long long int times_succ   = 0;
    float R = 0.4;
    Pythia pythia;
    Event& event = pythia.event;
    pythia.readString("Random:setSeed = on");
    pythia.readString("Random:Seed = 2019");
    pythia.readFile("./pythia.i");
    pythia.settings.listAll();
    pythia.init();

    // Set up tree and histogram
    // -------------------------
    
    //Initializes variables to be fed into tree
    Float_t mass_var,decay_length_var,pk_dca_var1,pk_dca_var2,trx_dca_var,pointing_angle_var,Drecon_dca_var,dca_angle_var,Z_jet_dmeson_var,paths_area_var,pk_p_frac_var,efficiency_pass_var,dmesonp_mag_var,jetp_mag_var;
    Int_t validation_var;
    
    //initialize counter variables for maintaining proportion of Dmeson containing jets to other jets
    Int_t a = 0,b = 0;
    
    //make the histograms for variables (mass in GeV, distance in mm, and angle in radians). Need to be resized/rebinned
    TFile* fout    = new TFile(fout_name.Data(),"RECREATE");
    TH1F* hmass_var = new TH1F("hmass_var","hmass_var",1000,0,4);
    TH1F* hdecay_length_var = new TH1F("hdecay_length_var","hdecay_length_var",1000,-2,2);
    TH1F* hpk_dca_var1 = new TH1F("hpk_dca_var1","hpk_dca_var1",1000,0,1000);
    TH1F* hpk_dca_var2 = new TH1F("hpk_dca_var2","hpk_dca_var2",1000,0,1000);
    TH1F* htrx_dca_var = new TH1F("htrx_dca_var","htrx_dca_var",10000,-2,2); 
    TH1F* hpointing_angle_var = new TH1F("hpointing_angle_var","hpointing_angle_var",1000,-5,5); 
    TH1F* hDrecon_dca_var = new TH1F("hDrecon_dca_var","hDrecon_dca_var",1000,-2,5);
    TH1F* hdca_angle_var = new TH1F("hdca_angle_var","hdca_angle_var",1000,0,1000);
    TH1F* hpaths_area_var = new TH1F("hpaths_area_var","hpaths_area_var",1000,0,1000);
    TH1F* hpk_p_frac_var = new TH1F("hpk_p_frac_var","hpk_p_frac_var",1000,0,1000);
    TH1F* hvalidation_var = new TH1F("hvalidation_var","hvalidation_var",3,-1,2);
    TH1F* hefficiency_pass_var = new TH1F("hefficiency_pass_var","hefficiency_pass_var",200,0,2);
    TH1F* hZ_jet_dmeson_var = new TH1F("hZ_jet_dmeson_var","hZ_jet_dmeson_var",1000,0,1000);
    TH1F* hmass_dist = new TH1F("hmass_dist","hmass_dist",200,0,2);
    TH1F* hdmesonp_mag_var = new TH1F("hdmesonp_mag_var","hdmesonp_mag_var",1000,0,1000);
    TH1F* hjetp_mag_var = new TH1F("hjetp_mag_var","hjetp_mag_var",1000,0,1000);
    
    //create tree for variables
    TTree* tree_train = new TTree("tree_train", "jet by jet");
    tree_train->Branch("mass_var", &mass_var, "mass_var/F");
    tree_train->Branch("decay_length_var", &decay_length_var, "decay_length_var/F");
    tree_train->Branch("pk_dca_var1", &pk_dca_var1, "pk_dca_var1/F");
    tree_train->Branch("pk_dca_var2", &pk_dca_var2, "pk_dca_var2/F");
    tree_train->Branch("trx_dca_var", &trx_dca_var, "trx_dca_var/F");
    tree_train->Branch("pointing_angle_var", &pointing_angle_var, "pointing_angle_var/F"); 
    tree_train->Branch("Drecon_dca_var", &Drecon_dca_var, "Drecon_dca_var/F");
    tree_train->Branch("dca_angle_var",&dca_angle_var,"dca_angle_var/F");
    tree_train->Branch("Z_jet_dmeson_var",&Z_jet_dmeson_var,"Z_jet_dmeson_var/F");
    tree_train->Branch("paths_area_var",&paths_area_var,"paths_area_var/F");
    tree_train->Branch("pk_p_frac_var",&pk_p_frac_var,"pk_p_frac_var/F");
    tree_train->Branch("validation_var", &validation_var, "validation_var/I");
    tree_train->Branch("efficiency_pass_var",&efficiency_pass_var,"efficiency_pass_var/F");
    tree_train->Branch("dmesonp_mag_var",&dmesonp_mag_var,"dmesonp_mag_var/F");
    tree_train->Branch("jetp_mag_var",&jetp_mag_var,"jetp_mag_var/F");
    
    // Event Loop
    // -----------
    
    //create array to store Dmeson identity data
    vector<int> darray = {0};
    darray.resize(10000000);
    int dcounter = 0;
    
    for (int iEvent = 0; iEvent < nEvents; ++iEvent) {
        
        //success counter
        if (!pythia.next()) {++times_failed; continue;}
        times_succ++;
        
        //intialize jet vector
        vector<PseudoJet> particles;
        int i_part = 0;
        
        // Particles loop
        // In this loop, we run through all the particles in a loop in order to:
        // 1: Designate which Dmesons produce pion kaon pairs
        // 2: Create Motherid
        // 3: store event-wide date on number of charm particles, etc
        // ----------------------------------------------------------
        for (unsigned int i = 0; i < event.size(); ++i) {
            if ( i<3 )continue;      // 0, 1, 2: total event and beams
            Particle&particle = event[i];
            
            //Finds indices of Dmesons with only 2 daughter particles, pion+ and kaon-.
            //--------------------------------------------------------
            
            //create vector containing daughter particles
            int d1,d2,d1partid,d2partid;
            bool dtrue1 = false;
            bool dtrue2 = false;
            bool kaon1 = false;
            bool pion1 = false;
            bool dnumber = false;
            vector<int> dlist = particle.daughterList();
            
            //see if decay profile matches D->pi+ka-
            if (particle.id() == 421){
                try{
                    //get id of (first) two daughters
                    d1 = dlist.at(0);
                    d2 = dlist.at(1);
                    Particle&d1part = event[d1];
                    Particle&d2part = event[d2];
                    d1partid = d1part.id();
                    d2partid = d2part.id();
                
                    //check number of daughters
                    if (dlist.size() == 2){
                        dnumber = true;
                    }
                    if (dlist.size()>2){
                        dnumber = false;
                    }
                
                    //check if first daughter fits decay profile
                    if ((d1partid == -321) || (d1partid == 211)){
                        dtrue1 = true;
                        if (d1partid == -321){
                            kaon1 = true;
                        }
                        if (d1partid == 211){
                            pion1 = true;
                        }
                    }
                    if ((d1partid != -321) && (d1partid != 211)){
                        dtrue1 = false;
                    }
                
                    //check if second daughter together with first daughter fits decay profile
                    if ((d2partid == -321) || (d2partid == 211)){
                        if (d2partid == -321 && pion1 == true){
                            dtrue2 = true;
                        }
                        if (d2partid == 211 && kaon1 == true){
                            dtrue2 = true;
                        }
                        else{
                            dtrue2 = false;
                        }
                    }
                    if ((d2partid != -321) && (d2partid != 211)){
                         dtrue2 = false;
                    }
                
                    //save as "good" Dmeson if decay profile is satisfied
                    if ((dnumber == true) && (dtrue1 == true) && (dtrue2 == true) && (particle.id() == 421)){
                        darray[dcounter] = i;
                        dcounter++;
                    } 
                }
            
                //if there are no or one daughters try loop will fail
                catch (const std::exception& e){ 
                }
            }
            
            //only regard final particles past this point
            if (!particle.isFinal()) continue;
            
            //creating mother id, index, particle id, and momentum for particle
            int pmothcheck = particle.mother1();
            int pmother_id = event[pmothcheck].id();
            Vec4 pv = particle.p();
            int pid = particle.id();
            
            //sort out possible unecessary particles
            if (abs(pid)==12 || abs(pid)==14 || abs(pid)==16) continue;
            if (abs(pid)>10000)
            cout<< pid << "exists" << endl;
            if (!particle.isCharged() && abs(pid)!=111 /*pi0*/ && abs(pid)!=22 /*photon*/ && abs(pid)!=311 /*K0*/ && abs(pid)!=2114 /*delta0*/ && abs(pid)!=130 /*K0L*/ && abs(pid)!=2112 /*n*/ && abs(pid)!=113 /*rho0*/ && abs(pid)==221 && abs(pid)==3122 && abs(pid) == 3212 && abs(pid)==421 && abs(pid)==3322 && abs(pid)==310){
                cout << pid << " is neutral and not taken into account" << endl;
            }
            
            //assign pythia information to particles in jet
            particles.push_back(PseudoJet());
            particles[i_part].reset_PtYPhiM(particle.pT(), particle.eta(), particle.phi());
            particles[i_part].set_user_index(654);//particle.id());
            particles[i_part].set_user_info(new MyInfo(particle.id(), particle.xProd(), particle.yProd(), particle.zProd(), particle.px(), particle.py(), particle.pz(), particle.charge(), particle.mother1(), pmother_id));
            ++i_part;
            
        }
        //create jets
        AreaDefinition area_def{active_area_explicit_ghosts, GhostedAreaSpec{2.0}};
        JetDefinition jet_def{antikt_algorithm, R};
        ClusterSequenceArea cs{particles, jet_def, area_def};
        vector<PseudoJet> jets_all = cs.inclusive_jets();
        Selector Fiducial_cut_selector = SelectorAbsEtaMax(1-R);
        vector<PseudoJet> jets = sorted_by_pt(Fiducial_cut_selector(jets_all));
        
        // Jets Loop, where we run over the jets in a particular event 
        //-------------------------------------------------
        for (int i = 0; i < jets.size(); ++i){
            
            //set jet and skip those with low pt and those composed entirely of "ghosts"
            PseudoJet& jet = jets[i];
            if (jet.is_pure_ghost()) continue;
            if (jet.pt()<5 ) continue;
            
            //initialize jet variables
            StThreeVector<float> jet_pt(0,0,0),jet_pts(0,0,0);
            Float_t jpx,jpy,jpz,jpxs,jpys,jpzs;
            vector<PseudoJet> constituents = jet.constituents();
            
            //smear jet pt
            TRandom3 rand;
            jpxs = rand.Gaus(jpx,((.5/100)*jpx+(.25/100)*pow(jpx,2)));
            jpys = rand.Gaus(jpy,((.5/100)*jpy+(.25/100)*pow(jpy,2)));
            jpzs = rand.Gaus(jpz,((.5/100)*jpz+(.25/100)*pow(jpz,2)));
            jet_pts = (jpxs,jpys,jpzs);
            
            //Record Jet pt and save magnitude
            jet_pt = jet.pt();
            jetp_mag_var = pow(jet_pt.x(),2)+pow(jet_pt.y(),2)+pow(jet_pt.z(),2);
            
            //Particle Loop: Loop over particles in a particular jet and consider oppositely charged pairs
            //--------------------------------------------------------------------------------------------
            
            for (unsigned int j = 0; j < constituents.size(); ++j) {
                //cut ghosts, neutral, and negative particles
                if(constituents[j].user_index()==-1) continue;
                long double jpcharge = constituents[j].user_info<MyInfo>().chrg_();
                if (jpcharge<=0) continue;
                
                
                //set particle id, decay vertex, and momentum from MyInfo
                int jpid = constituents[j].user_info<MyInfo>().pdg_id();
                
                //temporary bug-check
                if (jpid != 211) continue;
                
                long double jpvecx = constituents[j].user_info<MyInfo>().x_p();
                long double jpvecy = constituents[j].user_info<MyInfo>().y_p();
                long double jpvecz = constituents[j].user_info<MyInfo>().z_p();
                StThreeVector<long double> jtemp_p(jpvecx,jpvecy,jpvecz);
                long double jdecay_x = constituents[j].user_info<MyInfo>().x_Dec();
                long double jdecay_y = constituents[j].user_info<MyInfo>().y_Dec();
                long double jdecay_z = constituents[j].user_info<MyInfo>().z_Dec();
                StThreeVector<long double> jtemp_o(jdecay_x,jdecay_y,jdecay_z);
                const long double jB = 0.5*tesla;
                
                //create preliminary helix for smear
                const long double jtemp_B = 0.5*tesla;
                long double jtemp_q = constituents[j].user_info<MyInfo>().chrg_();
                StPhysicalHelix jtemp_helix(jtemp_p, jtemp_o, jtemp_B, jtemp_q);
                long double min = 10000000.0;
                
                //set origin for smeared helix
                StThreeVector<long double> jo=jtemp_o;
                
                //take it out to the 500mm sphere (position detector begins)
                
                long double hit_y;
                for (long double y = 1000; y > -1000; --y) {
                    StThreeVector<long double> jo_pos = jtemp_helix.at(y);
                    long double jdist = fabs(pow(jo_pos.x(),2)+pow(jo_pos.y(),2)+pow(jo_pos.z(),2)-2500);
                    if (jdist<min){
                         min = jdist;
                        hit_y = y;
                    }
                }
                
                //set momenta and position
                jo = jtemp_helix.at(hit_y);
                StThreeVector<long double> momenta = jtemp_helix.momentumAt(hit_y,jB);
                jpvecx = momenta.x();
                jpvecy = momenta.y();
                jpvecz = momenta.z();
                
                //actual smear
                jpvecx = rand.Gaus(jpvecx,((.5/100)*jpvecx+(.25/100)*pow(jpvecx,2)));
                jpvecy = rand.Gaus(jpvecy,((.5/100)*jpvecy+(.25/100)*pow(jpvecy,2)));
                jpvecz = rand.Gaus(jpvecz,((.5/100)*jpvecz+(.25/100)*pow(jpvecz,2)));
                
                
                //assign pion mass to positively charged particle
                long double m_1 = .1395701835;
                
                //create smeared trajectory helix
                StThreeVector<long double> jp(jpvecx,jpvecy,jpvecz);
                long double jq = constituents[j].user_info<MyInfo>().chrg_();
                StPhysicalHelix jhelix(jp, jo, jB, jq);
                StThreeVector<long double> pv(0,0,0);
                    
                //find dca from helix to primary vertex
                long double kpdca1 = jhelix.distance(pv);
                    
                //get mother id and index
                int jmothindex1 = constituents[j].user_info<MyInfo>().moth_1();
                int jmothid = constituents[j].user_info<MyInfo>().mother_id();
                    
                //second particle loop
                for (unsigned int k = 1; k < constituents.size(); ++k) {
                    
                    //cut ghosts, neutral, and positive particles
                    if(constituents[k].user_index()==-1) continue;
                    long double kpcharge = constituents[k].user_info<MyInfo>().chrg_();
                    if (kpcharge>=0) continue;
                        
                    //set particle id, decay vertex, and momentum from MyInfo
                    int kpid = constituents[k].user_info<MyInfo>().pdg_id();
                    
                    //temporary bug-check
                    if (kpid != -321) continue;
                    
                    long double kpvecx = constituents[k].user_info<MyInfo>().x_p();
                    long double kpvecy = constituents[k].user_info<MyInfo>().y_p();
                    long double kpvecz = constituents[k].user_info<MyInfo>().z_p();
                    long double kdecay_x = constituents[k].user_info<MyInfo>().x_Dec();
                    long double kdecay_y = constituents[k].user_info<MyInfo>().y_Dec();
                    long double kdecay_z = constituents[k].user_info<MyInfo>().z_Dec();
                    StThreeVector<long double> ktemp_p(kpvecx,kpvecy,kpvecz);
                    StThreeVector<long double> ktemp_o(kdecay_x,kdecay_y,kdecay_z);
                    
                    //create preliminary helix for smear
                    const long double ktemp_B = 0.5*tesla;
                    long double ktemp_q = constituents[j].user_info<MyInfo>().chrg_();
                    StPhysicalHelix ktemp_helix(ktemp_p, ktemp_o, ktemp_B, ktemp_q);
                    min = 10000000.0;
                    
                    //set origin for smeared helix
                    StThreeVector<long double> ko;
                        
                    //take it out to the 500mm sphere (position detector begins)
                    
                    for (long double y = 1000; y > -1000; --y) {
                        StThreeVector<long double> ko_pos = ktemp_helix.at(y);
                        long double kdist = fabs(pow(ko_pos.x(),2)+pow(ko_pos.y(),2)+pow(ko_pos.z(),2)-2500);
                        //taking it out to the 500mm sphere
                        if (kdist<min){
                            min = kdist;
                            hit_y = y;
                        }
                    }
                    
                    //set momenta and position
                    ko = ktemp_helix.at(hit_y);
                    StThreeVector<long double> momenta = ktemp_helix.momentumAt(hit_y,jB);
                    kpvecx = momenta.x();
                    kpvecy = momenta.y();
                    kpvecz = momenta.z();
                    
                    //actual smear
                    kpvecx = rand.Gaus(kpvecx,((.5/100)*kpvecx+(.25/100)*pow(kpvecx,2)));
                    kpvecy = rand.Gaus(kpvecy,((.5/100)*kpvecy+(.25/100)*pow(kpvecy,2)));
                    kpvecz = rand.Gaus(kpvecz,((.5/100)*kpvecz+(.25/100)*pow(kpvecz,2)));
                        
                    //assign kaon mass to negatively charged particle
                    long double m_2 = .493677;
                    
                    //create smeared trajectory helix
                    StThreeVector<long double> kp(kpvecx,kpvecy,kpvecz);
                    float kB = ktemp_B;
                    float kq = ktemp_q;
                    StPhysicalHelix khelix(kp, ko, kB, kq);
                            
                    //find dca from helix to primary vertex
                    long double kpdca2 = khelix.distance(pv);
                            
                    //gives pathlengths at which helices attain distance of closest approach
                    pair<long double,long double> s = jhelix.pathLengths(khelix);
                            
                    //Calculate variables
                    //-------------------
                    
                    //-----------------------------------------------//
                    //0, Invariant Mass. Calculated to verify results//
                    //-----------------------------------------------//
                            
                    //calculate mass using E = K + Eo
                    StThreeVector<float> j_p = jhelix.momentumAt(s.first,kB);
                    StThreeVector<float> k_p = khelix.momentumAt(s.second,kB);
                    mass_var =(sqrt(pow(m_1,2)+pow(m_2,2)+2*(sqrt(pow(m_1,2)+((j_p.x()*j_p.x())+(j_p.y()*j_p.y())+(j_p.z()*j_p.z())))*sqrt(pow(m_2,2)+((k_p.x()*k_p.x())+(k_p.y()*k_p.y())+(k_p.z()*k_p.z())))-(j_p.x()*k_p.x())-(j_p.y()*k_p.y())-(j_p.z()*k_p.z()))));
                    hmass_var->Fill(mass_var);
                    
                            
                    //---------------------------------------------------------------------//
                    //1, Decay Length from Primary Vertex to reconstructed secondary vertex//
                    //---------------------------------------------------------------------//
                            
                    //take average position of daughter helices at DCA pathlengths, approximates secondary vertex
                    StThreeVector<float> vecdlen =((jhelix.at(s.first)+khelix.at(s.second))/2);
                            
                    //find magnitude of distance
                    Float_t dlen = sqrt(pow(vecdlen.x(),2)+pow(vecdlen.y(),2)+pow(vecdlen.z(),2));
                    hdecay_length_var->Fill(dlen);
                    decay_length_var = dlen;
                            
                    //---------------------------------------------------------//
                    //2 and 3, DCAs to primary vertex from pion and kaon tracks//
                    //---------------------------------------------------------//
                            
                    hpk_dca_var1->Fill(kpdca1);
                    hpk_dca_var2->Fill(kpdca2);
                    pk_dca_var1 = kpdca1;
                    pk_dca_var2 = kpdca2;
                            
                    //--------------------------------------//
                    //4 DCA between the pion and kaon tracks//
                    //--------------------------------------//
                            
                    Float_t trxdca = abs(jhelix.at(s.first)-khelix.at(s.second)); 
                    htrx_dca_var->Fill(trxdca);
                    trx_dca_var = trxdca;
                            
                    //------------------------------------------------------------------------------------------//
                    //5 Pointing angle between p vector of reconstructed mother particle and decay length vector//
                    //------------------------------------------------------------------------------------------//
                            
                    //conservation of momentum, momentum of daughters at decay vertex = momentum of mother
                    StThreeVector<float> jhelixp = jhelix.momentumAt(s.first,kB);
                    StThreeVector<float> khelixp = khelix.momentumAt(s.second,kB);
                    StThreeVector<float> dmesonp = jhelixp + khelixp;
                            
                    //calculate angle
                    float dotprod = ((dmesonp.x()*vecdlen.x())+(dmesonp.y()*vecdlen.y())+(dmesonp.z()*vecdlen.z()));
                    float dmesonp_mag = sqrt(pow(dmesonp.x(),2)+pow(dmesonp.y(),2)+pow(dmesonp.z(),2));
                    float vecdlen_mag = sqrt(pow(vecdlen.x(),2)+pow(vecdlen.y(),2)+pow(vecdlen.z(),2));
                    float distprod = dmesonp_mag*vecdlen_mag;
                    float cosalpha = dotprod/distprod;
                    Float_t pdpa = acos(dotprod/distprod);
                            
                    //if both particles decayed at or very near primary vertex, set angle to pi (will be nan otherwise)
                    if ((fabs(vecdlen.x())+fabs(vecdlen.y())+fabs(vecdlen.z())) < 1e-35){
                        pdpa = 3.14159265359;
                    }
                    //bug fix
                    if (((pdpa>0) == false)&&((pdpa<0) == false)&&((pdpa=0) == false)){
                        pdpa = 0;
                    }
                    hpointing_angle_var->Fill(pdpa);
                    pointing_angle_var = pdpa;
                            
                    //saves magnitude of reconstructed D meson momentum
                    dmesonp_mag_var = dmesonp_mag;
                            
                    //------------------------------------------------//
                    //6 DCA of reconstructed D meson to primary vertex//
                    //------------------------------------------------//
                            
                    //creates decay helix for reconstructed D meson
                    StThreeVector<float> lp = dmesonp;
                    StThreeVector<float> lo = vecdlen;
                    const double lB = 0.5*tesla;
                    double lq = 0.0;
                    StPhysicalHelix dhelix(lp, lo, lB, lq);
                            
                    //find distance
                    StThreeVector<float> origin = (0,0,0);
                    float t = dhelix.distance(origin);
                    Float_t dpvdca = t;
                    hDrecon_dca_var->Fill(dpvdca);
                    Drecon_dca_var = dpvdca;
                            
                    //----------------------------------//
                    //7 Angle between two DCA candidates//
                    //----------------------------------//
                            
                    //calculate angle, check "StHelix.cc" for jpathLengths
                    pair<float,float> r = jhelix.jpathLengths(khelix);
                    StThreeVector<float> vecdlen2 =((jhelix.at(r.first)+khelix.at(r.second))/2);
                    dca_angle_var = vecdlen.angle(vecdlen2);
                            
                    //bug fix
                    if ((!(dca_angle_var<0))&&(!(dca_angle_var>=0))){
                        dca_angle_var=0.0;
                    }
                            
                    //------------------------------------//
                    //8 Z ,jet momentum & D meson momentum//
                    //------------------------------------//
                            
                    Z_jet_dmeson_var = dmesonp_mag_var/jetp_mag_var;
                            
                    //------------------------------------------------------//
                    //9, Path volume (not a true area, but an approximation)//
                    //------------------------------------------------------//
                            
                    //calculates distance between points along tracks incremented by 1cm of pathlength
                    for (unsigned int j = 0; j < 100; ++j) {
                        StThreeVector<float> jpos = jhelix.at(s.first);
                        StThreeVector<float> kpos = khelix.at(s.second);
                        paths_area_var = paths_area_var+sqrt(pow(jpos.x()-kpos.x(),2)+pow(jpos.y()-kpos.y(),2)+pow(jpos.z()-kpos.z(),2));
                    }
                            
                    //-----------------------------------//
                    //10 Momentum fraction pion over kaon//
                    //-----------------------------------//
                            
                    pk_p_frac_var = (sqrt(pow(jhelixp.x(),2)+pow(jhelixp.y(),2)+pow(jhelixp.z(),2)))/(sqrt(pow(khelixp.x(),2)+pow(khelixp.y(),2)+pow(khelixp.z(),2)));
                    //-------------------//
                    //11, Efficiency pass//
                    //-------------------//
                            
                    //generate random number, and give a "disregard" marker to 20% of particles
                    float rndm = rand.Uniform(1);
                    if (rndm >= .8){
                        efficiency_pass_var = 0;
                    }
                    if (rndm < .8){
                        efficiency_pass_var = 1;
                    }
                            
                    //----------------------//
                    //12 Validation variable//
                    //----------------------//
                            
                    //find mother index and id of second particle
                    int kmothindex1 = constituents[k].user_info<MyInfo>().moth_1();
                    int kmothid = constituents[k].user_info<MyInfo>().mother_id();
                    int truevar = 0;
                            
                    //check if pair shares a mother and if mother is a "good" D-meson (following pi+ka- decay profile)
                    bool dmeson = false;
                    if (jmothindex1 == kmothindex1){
                        for (int q = 0; q < darray.size(); ++q) {
                            if (darray[q] == jmothindex1){
                                dmeson = true;
                            }
                            else continue;
                        }
                    }
                    
                    
                    //check daughters
                    Particle& particle_ = event[jmothindex1];
                    vector<int> dist = particle_.daughterList();
                    
                    //determine if the pair represents the daughters of desired decay profile
                    if (jmothindex1 == kmothindex1 && jmothid == 421 && kmothid == 421 && dmeson == true && dist.size()==2){
                        truevar = 1;
                    }
                    else{
                        truevar = 0;
                    }
                        
                    //fill into tree with validation variable indicating whether there was a "good" D-meson or not
                    if (truevar == 1){
                        validation_var = 1;
                        tree_train->Fill();
                        hvalidation_var->Fill(truevar);
                        a = a+1;
                    }
                    if ((truevar == 0) && (b < a)){
                        validation_var = 0;
                        tree_train->Fill();
                        hvalidation_var->Fill(truevar);
                        b = b+1;
                    }
                            
                    //---------------------------------------//
                    //Fill the mass of particles in cutspace//
                    //---------------------------------------//
                    if ((decay_length_var<8)&&(decay_length_var>=0.0)&&(pk_dca_var1<.5)&&(pk_dca_var1>=0.0)&&(pk_dca_var2<.5)&&(pk_dca_var2>=0.0)&&(trx_dca_var<.0000000006)&&(trx_dca_var>=0.0)&&(pointing_angle_var<.04)&&(pointing_angle_var>=0.0)&&(Drecon_dca_var<10.0)&&(Drecon_dca_var>0.0)){
                        hmass_dist->Fill(mass_var);
                    }
                }
            }//end of constituents loop
        }//end of jets loop
    }//end of event loop
    cout << "Times failed    " << times_failed << endl;
    cout << "Times succeeded " << times_succ << endl;
    fout->Write();
}
