#include <iostream>
#include <vector>
#include <utility>
#include <chrono>
#include <ctime>
#include "ROOT/RDataFrame.hxx"
#include "ROOT/RVec.hxx"
#include "Math/Vector4D.h"


using RFilter = ROOT::RDF::RInterface<ROOT::Detail::RDF::RJittedFilter>;

std::map<std::string, float> cross_section = {
    {"GluGluToHToTauTau", 19.6}, {"VBF_HToTauTau", 1.55}, {"DYJetsToLL", 3503.7},
    {"TTbar", 225.2}, {"W1JetsToLNu", 6381.2}, {"W2JetsToLNu", 2039.8}, {"W3JetsToLNu", 612.5},
    {"Run2012B_TauPlusX", 1.0}, {"Run2012C_TauPlusX", 1.0},
};

template <typename T>
RFilter muon_selection(T &d)
{
    auto d_temp = d.Filter("nMuon>0 & HLT_IsoMu17_eta2p1_LooseIsoPFTau20==1")
                   .Define("goodMuons", "(abs(Muon_eta)<2.1) & (Muon_pt>17) & (Muon_tightId==1)")
                   .Filter("ROOT::VecOps::Sum(goodMuons)>0");
    return d_temp;
}

template <typename T>
RFilter tau_selection(T &d)
{
    auto d_temp = d.Filter("nTau>0 & HLT_IsoMu17_eta2p1_LooseIsoPFTau20==1")
                   .Define("goodTaus", "Tau_charge!=0 & abs(Tau_eta)<2.3 & Tau_pt>20 & Tau_idDecayMode==1 & Tau_idIsoTight==1 & Tau_idAntiEleTight==1 & Tau_idAntiMuTight==1")
                   .Filter("ROOT::VecOps::Sum(goodTaus)>0");
    return d_temp;
}

template <typename T>
RFilter take_jet(T &d)
{
    auto first_jet = [] (ROOT::RVec<unsigned long> &goodJets_idx, ROOT::RVec<float> &jet_var)
    {
        auto var = ROOT::VecOps::Take(jet_var, goodJets_idx);
        if (goodJets_idx.size()>0)
            return var[0];
        return -999.f;
    };
    
    auto second_jet = [] (ROOT::RVec<unsigned long> &goodJets_idx, ROOT::RVec<float> &jet_var)
    {
        auto var = ROOT::VecOps::Take(jet_var, goodJets_idx);
        if (goodJets_idx.size()>1)
            return var[1];
        return -999.f;
    };
    
    auto compute_mjj = [] (int nGoodJets, ROOT::Math::PtEtaPhiMVector jjp4)
    {
        if (nGoodJets>1) {
            return float(jjp4.M());
        }
        return -999.f;
    };
    
    auto compute_ptjj = [] (int nGoodJets, ROOT::Math::PtEtaPhiMVector jjp4)
    {
        if (nGoodJets>1) {
            return float(jjp4.Pt());
        }
        return -999.f;
    };
    
    auto compute_jdeta = [](int nGoodJets, float jeta_1, float jeta_2){
        if (nGoodJets>1) {
            return jeta_1 - jeta_2;
        }
        return -999.f;
    };
    
    auto d_temp = d.Define("goodJets", "Jet_pt>30 & abs(Jet_eta)<4.7 & Jet_puId==1")
                   .Define("goodJets_idx", "ROOT::VecOps::Nonzero(goodJets)")
                   .Define("jpt_1", first_jet, {"goodJets_idx", "Jet_pt"})
                   .Define("jpt_2", second_jet, {"goodJets_idx", "Jet_pt"})
                   .Define("jeta_1", first_jet, {"goodJets_idx", "Jet_eta"})
                   .Define("jeta_2", second_jet, {"goodJets_idx", "Jet_eta"})
                   .Define("jphi_1", first_jet, {"goodJets_idx", "Jet_phi"})
                   .Define("jphi_2", second_jet, {"goodJets_idx", "Jet_phi"})
                   .Define("jm_1", first_jet, {"goodJets_idx", "Jet_mass"})
                   .Define("jm_2", second_jet, {"goodJets_idx", "Jet_mass"})
                   .Define("jbtag_1", first_jet, {"goodJets_idx", "Jet_btag"})
                   .Define("jbtag_2", second_jet, {"goodJets_idx", "Jet_btag"})
                   .Define("nGoodJets", "int(goodJets_idx.size())")
                   .Define("jp4_1", "ROOT::Math::PtEtaPhiMVector(jpt_1, jeta_1, jphi_1, jm_1)")
                   .Define("jp4_2", "ROOT::Math::PtEtaPhiMVector(jpt_2, jeta_2, jphi_2, jm_2)")
                   .Define("jjp4", "ROOT::Math::PtEtaPhiMVector(jpt_1, jeta_1, jphi_1, jm_1)")
                   .Define("jj_m", compute_mjj, {"nGoodJets", "jjp4"})
                   .Define("jj_pt", compute_ptjj, {"nGoodJets", "jjp4"})
                   .Define("jj_delta", compute_jdeta, {"nGoodJets", "jeta_1", "jeta_2"});
    return d_temp;
}

template <typename T>
RFilter h2mutau(T &d)
{
    auto goodPair = [] (ROOT::RVec<int> &goodMuons, ROOT::RVec<int> &goodTaus, ROOT::RVec<float> &eta1, ROOT::RVec<float> &eta2, ROOT::RVec<float> &phi1, ROOT::RVec<float> &phi2, ROOT::RVec<float> &pt1, ROOT::RVec<float>& iso2, ROOT::RVec<int>& charge1, ROOT::RVec<int>& charge2)
    {
        auto muons_idx = ROOT::VecOps::Nonzero(goodMuons);
        auto taus_idx = ROOT::VecOps::Nonzero(goodTaus);
        if(taus_idx.size() == 1 & muons_idx.size() == 1){
            auto DeltaR = ROOT::VecOps::DeltaR(eta1[muons_idx[0]], eta2[taus_idx[0]], phi1[muons_idx[0]], phi2[taus_idx[0]]);
            if(DeltaR > 0.5){
                return std::pair<int, int>(muons_idx[0], taus_idx[0]);}
            return std::pair<int, int>(-999, -999);
        }
        else{
            auto idx_pt_max = ROOT::VecOps::ArgMax(pt1);
            auto idx_iso_min = ROOT::VecOps::ArgMin(iso2);
            auto DeltaR = ROOT::VecOps::DeltaR(eta1[idx_pt_max], eta2[idx_iso_min], phi1[idx_pt_max], phi2[idx_iso_min]);
            if(DeltaR > 0.5){
                return std::pair< int , int >(idx_pt_max, idx_iso_min);}
            return std::pair<int, int>(-999, -999);}
        
    };
    auto d_temp = d.Define("h2mutau", goodPair, {"goodMuons", "goodTaus", "Muon_eta", "Tau_eta", "Muon_phi", "Tau_phi",                                           "Muon_pt", "Tau_relIso_all", "Muon_charge", "Tau_charge"})
                   .Define("idx_muon", "h2mutau.first").Filter("idx_muon >= 0")
                   .Define("idx_tau", "h2mutau.second");
    return d_temp;
}

template <typename T>
RFilter ML_variables(T &d)
{
    auto compute_mt = [](float pt, float phi, float met_pt, float met_phi){
        double dphi = ROOT::VecOps::DeltaPhi(phi, met_phi);
        return std::sqrt(2.0 * pt * met_pt * (1.0 - std::cos(dphi)));
    };
    
    auto compute_DeltaR = [](float mu_eta, float tau_eta, float mu_phi, float tau_phi){
        return ROOT::VecOps::DeltaR(mu_eta, tau_eta, mu_phi, tau_phi);
    };
    
    auto d_temp = d.Define("muon_pt", "Muon_pt[idx_muon]")
                   .Define("muon_eta", "Muon_eta[idx_muon]")
                   .Define("muon_phi", "Muon_phi[idx_muon]")
                   .Define("muon_m", "Muon_mass[idx_muon]")
                   .Define("muon_charge", "Muon_charge[idx_muon]")
                   .Define("muon_p4", "ROOT::Math::PtEtaPhiMVector(muon_pt, muon_eta, muon_phi, muon_m)")
                   .Define("muon_iso", "Muon_pfRelIso03_all[idx_muon]")
                   .Define("tau_pt", "Tau_pt[idx_tau]")
                   .Define("tau_eta", "Tau_eta[idx_tau]")
                   .Define("tau_phi", "Tau_phi[idx_tau]")
                   .Define("tau_m", "Tau_mass[idx_tau]")
                   .Define("tau_charge", "Tau_charge[idx_muon]")
                   .Define("tau_iso", "Tau_relIso_all[idx_tau]")
                   .Define("tau_p4", "ROOT::Math::PtEtaPhiMVector(tau_pt, tau_eta, tau_phi, tau_m)")
                   .Define("p4", "muon_p4+tau_p4")
                   .Define("m_vis", "p4.M()")
                   .Define("pt_vis", "p4.Pt()")
                   .Define("mt_mu", compute_mt, {"muon_pt", "muon_phi", "MET_pt", "MET_phi"})
                   .Define("mt_tau", compute_mt, {"tau_pt", "tau_phi", "MET_pt", "MET_phi"})
                   .Define("dRmu_tau", compute_DeltaR, {"muon_eta", "tau_eta", "muon_phi", "tau_phi"});
    
    return d_temp;
}

std::vector<std::string> finalVariables = {
    "nGoodJets", "PV_npvs", "dRmu_tau",
    "muon_pt", "muon_eta", "muon_phi", "muon_m", "muon_iso", "mt_mu", "muon_charge",  // muon variables
    "tau_pt", "tau_eta", "tau_phi", "tau_m", "tau_iso", "mt_tau", "tau_charge", // tau varibales
    "jpt_1", "jeta_1", "jphi_1", "jm_1", "jbtag_1", // leading jet variables
    "jpt_2", "jeta_2", "jphi_2", "jm_2", "jbtag_2", // trial jet variables
    "MET_pt", "MET_phi", "m_vis", "pt_vis", "jj_m", "jj_pt", "jj_delta", //high level variables
    "weight"
};

int main()
{
    ROOT::EnableImplicitMT();
    int poolSize = ROOT::GetImplicitMTPoolSize();
    std::cout << "Pool size used for multi-threading: " << poolSize << std::endl;
    
    std::vector<std::string> ROOT_files = {
        "GluGluToHToTauTau",
        "VBF_HToTauTau",
        "DYJetsToLL",
        "TTbar",
        "W1JetsToLNu", "W2JetsToLNu", "W3JetsToLNu", //MC
        "Run2012B_TauPlusX", "Run2012C_TauPlusX" // Data
    };
    
    auto start = std::chrono::system_clock::now();
    for (const std::string &sample : ROOT_files) {
        std::cout << "***********************************************************************************" << std::endl;
        std::cout << "Processing sample: " << sample << std::endl;
        ROOT::RDataFrame d_root("Events", sample + ".root");
    
        int num_events = *d_root.Count();
        const float integratedLuminosity = 11.467 * 1000.0; //(pb^-1)
        float weight = 1.;
        if(sample.find("Run") != 0)
            weight = (cross_section[sample]/num_events) * integratedLuminosity;
        auto d_MCweight = d_root.Define("weight", [weight](){ return weight;});
    
        std::cout << "\t Number of events in ROOT file: " << num_events << std::endl;
        auto d_muon = muon_selection(d_MCweight);
        std::cout << "\t Number of events after muon selection: " << *d_muon.Count() << std::endl;
        auto d_tau = tau_selection(d_muon);
        std::cout << "\t Number of events after tau selection: " << *d_tau.Count() << std::endl;
        auto d_h2mutau = h2mutau(d_tau);
        std::cout << "\t Number of events after selecting candidates for the Higgs (H->ττ): " << *d_h2mutau.Count() << std::endl;
        auto d_variables = ML_variables(d_h2mutau);
        auto d_jets = take_jet(d_variables);
        auto d = d_jets.Filter("muon_charge*tau_charge<0");
        d.Snapshot("Events", sample + "_selected.root", finalVariables);
    }
    
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Finished computation at " << std::ctime(&end_time) << "Total elapsed time: " << elapsed_seconds.count() << "s" << std::endl;
    
}
