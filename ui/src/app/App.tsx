import { useState } from "react";
import { EchoSidebar } from "./components/echo-sidebar";
import { EchoHeader } from "./components/echo-header";
import { HomePage } from "./components/pages/home-page";
import { MeetingNotesPage } from "./components/pages/meeting-notes-page";
import { MeetingDetailPage } from "./components/pages/meeting-detail-page";
import { TemplatesPage } from "./components/pages/templates-page";
import { SettingsPage } from "./components/pages/settings-page";
import { SearchPalette } from "./components/search-palette";
import { ActivityDrawer } from "./components/activity-drawer";

export type PageId = "home" | "notes" | "detail" | "templates" | "settings";

const breadcrumbs: Record<PageId, string> = {
  home: "Home",
  notes: "Meeting Notes",
  detail: "Meeting Notes / Detail",
  templates: "Templates",
  settings: "Settings",
};

export default function App() {
  const [page, setPage] = useState<PageId>("home");
  const [selectedMeetingId, setSelectedMeetingId] = useState<string | null>(null);
  const [searchOpen, setSearchOpen] = useState(false);
  const [activityOpen, setActivityOpen] = useState(false);

  const openMeeting = (meetingId?: string) => {
    if (meetingId) {
      setSelectedMeetingId(meetingId);
    }
    setPage("detail");
  };

  return (
    <div className="size-full min-h-screen flex bg-echo-bg text-echo-text" style={{ fontFamily: "Inter, ui-sans-serif, system-ui, -apple-system, sans-serif" }}>
      <EchoSidebar current={page} onNavigate={setPage} />
      <div className="flex-1 min-w-0 flex flex-col">
        <EchoHeader
          breadcrumb={breadcrumbs[page]}
          onOpenSearch={() => setSearchOpen(true)}
          onOpenActivity={() => setActivityOpen(true)}
          attentionCount={1}
        />
        <main className="flex-1 overflow-y-auto">
          <div className="max-w-[1320px] mx-auto px-8 py-6">
            {page === "home" && <HomePage onOpenReview={openMeeting} />}
            {page === "notes" && <MeetingNotesPage onOpen={openMeeting} />}
            {page === "detail" && <MeetingDetailPage meetingId={selectedMeetingId} onBack={() => setPage("notes")} />}
            {page === "templates" && <TemplatesPage />}
            {page === "settings" && <SettingsPage />}

            <div className="text-center text-[11px] text-echo-text-faint py-6">
              ECHO · Executive Calls, Highlights & Outcomes
            </div>
          </div>
        </main>
      </div>

      <SearchPalette open={searchOpen} onClose={() => setSearchOpen(false)} onOpenMeeting={openMeeting} />
      <ActivityDrawer open={activityOpen} onClose={() => setActivityOpen(false)} />
    </div>
  );
}
